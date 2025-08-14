import gc
import torch
import json
import yaml
import sys
import time
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict
import click
from loguru import logger
from transformers import AutoTokenizer, AutoAdapterModel
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, classification_report
from transformers import TrainingArguments, AdapterTrainer, PfeifferConfig, PfeifferInvConfig, LoRAConfig, IA3Config
import torch.nn as nn
from transformers.adapters import PredictionHead, CausalLMHead
from tqdm import tqdm


@click.group()
def cli():
    pass


@cli.command()
@click.argument('path', required=True)
def run(path):
    click.echo('running experiments')
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    run_all_experiments(path)


def compute_metrics(a, p):
    return {"accuracy": (p == a).mean(),
            "f1": f1_score(a, p, average='macro'),
            "precision": precision_score(a, p, average='macro'),
            "recall": recall_score(a, p, average='macro')}


def encode_batch(batch, tokenizer, max_length):
    return tokenizer(batch["text"], max_length=max_length, truncation=True, add_special_tokens=False)


def insert_list(target_list, position, new_elements, extend_with, extend_to=512):
    copied_list = target_list.copy()
    for index, item in enumerate(new_elements):
        copied_list.insert(position+index, item)
    copied_list.extend([extend_with]*(extend_to-len(copied_list)))
    return copied_list


def extend_attention_mask(target_list, pattern_length, extend_to=512):
    copied_list = target_list.copy()
    copied_list.extend([1]*pattern_length)
    copied_list.extend([0]*(extend_to-len(copied_list)))
    return copied_list


class PEThead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        id2tokenid,
        vocab_size=None,
        **kwargs
    ):
        super().__init__(head_name)
        self.config = {
            "vocab_size": model.config.vocab_size,
            "id2tokenid": {key:id2tokenid[key] for key in sorted(id2tokenid)}, # ensures sorted dict
            "id2tokenid_values": sorted(set([value for sublist in id2tokenid.values() for value in sublist])),
        }
        self.build(model)

    def build(self, model):
        model_config = model.config
        # Additional FC layers
        pred_head = []
        pred_head.append(nn.Linear(model_config.hidden_size, model_config.hidden_size))
        pred_head.append(nn.GELU())
        pred_head.append(nn.LayerNorm(model_config.hidden_size, eps=1e-12))
        for i, module in enumerate(pred_head):
            self.add_module(str(i), module)

        # Final embedding layer
        self.add_module(
            str(len(pred_head)),
            nn.Linear(model_config.hidden_size, len(self.config["id2tokenid_values"]), bias=True),
        )

        self.apply(model._init_weights)
        self.train(model.training)  # make sure training mode is consistent


    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        # First, pass through all layers except the last embedding layer
        seq_outputs = outputs[0]
        for i in range(len(self) - 1):
            seq_outputs = self[i](seq_outputs)

        # Now, pass through an invertible adapter if available
        inv_adapter = kwargs.pop("invertible_adapter", None)
        if inv_adapter is not None:
            seq_outputs = inv_adapter(seq_outputs, rev=True)

        # Finally, pass through the last embedding layer
        lm_logits = self[len(self) - 1](seq_outputs)

        loss = None
        loss_fct = nn.CrossEntropyLoss()
        labels = kwargs.pop("labels", None)
        n_mask_token = max([len(self.config["id2tokenid"][i]) for i in range(len(self.config["id2tokenid"]))])
        id2newid = {i: z for i, z in
                    zip(self.config["id2tokenid_values"], range(len(self.config["id2tokenid_values"])))}
        id2dim = {k: [id2newid[v1] for v1 in v] for k, v in self.config["id2tokenid"].items()}
        verbalizerid = list(id2dim.values())

        #mask_indices = []
        #for i in range(n_mask_token):
        #    mask_indices["mask_indices"+str(i+1)] = kwargs.get("mask_indices"+str(i+1))

        mask_indices = kwargs.get("mask_indices1")
        logits_mask = lm_logits[range(lm_logits.shape[0]), mask_indices, :]
        logits_for_loss = logits_mask[:, [k[0] for k in verbalizerid]]
        for i in range(n_mask_token-1):
            mask_indices = kwargs.get("mask_indices"+str(i+2))
            logits_mask = lm_logits[range(lm_logits.shape[0]), mask_indices, :]
            logits_for_loss += logits_mask[:, [k[i+1] for k in verbalizerid]]

        #logits_mask1 = lm_logits[range(lm_logits.shape[0]),mask_indices1,:][:,[k[0] for k in verbalizerid]]
        #logits_mask2 = lm_logits[range(lm_logits.shape[0]),mask_indices2,:]

        #id2newid = {i:z for i,z in zip(self.config["id2tokenid_values"], range(len(self.config["id2tokenid_values"])))}
        #id2dim = {k: [id2newid[v1] for v1 in v] for k, v in self.config["id2tokenid"].items()}
        #verbalizerid = list(id2dim.values())

        #logits_mask1 = logits_mask1[:,[k[0] for k in verbalizerid]]
        #logits_mask2 = logits_mask2[:,[k[1] for k in verbalizerid]]

        #logits_for_loss = logits_mask1 #+ logits_mask2

        if labels is not None:
            loss = loss_fct(logits_for_loss.view(-1, len(self.config["id2tokenid"])), labels.view(-1))
        outputs = (logits_for_loss,) + outputs[1:]
        if loss is not None:
            outputs = (loss,) + outputs
        return outputs


def read_config(path):
    if type(path) == str:
        path = Path(path)
    with open (path / 'config.yml', 'r', encoding='utf8') as cfg:
        config = yaml.safe_load(cfg)
    logger.debug('loaded config')
    if not isinstance(config['adapter']['arch'], list):
        config['adapter']['arch'] = [config['adapter']['arch']]
    if 'test_path' not in config['dataset'].keys():
        config['dataset']['test_path'] = 'as_path'

    if 'per_device_train_batch_size' not in config['adapter']:
        config['adapter']['per_device_train_batch_size'] = 2
    return config


def find_experiments(path):
    if type(path) == str:
        path = Path(path)
    for experiment in path.iterdir():
        if experiment.is_dir() and (experiment / 'train.csv').exists():
            yield experiment
        else:
            yield from find_experiments(experiment)







def data_from_csv(path, pattern, verbalizer, tokenizer, labels=False, split_name='test'):
    """creates a dataset for training or prediction from a csv file.
    The file needs to have two columns, first with text, second with labels
    """
    dataset = pd.read_csv(path, header=None, names=['text', 'label'])
    data = {}
    return_labels = False
    if not labels:
        return_labels = True
        labels = list(dataset.label.unique())
        labels.sort()
        id2tokenid = {idx:tokenizer(verbalizer[label], add_special_tokens=False, return_attention_mask=False).input_ids[0:pattern['n_mask_token']] for idx,label in enumerate(labels)}
    dataset['labels'] = [labels.index(x) for x in dataset['label']]
    dict_name = 'eval' if split_name == 'test' else 'train'
    dataset_dict = DatasetDict({dict_name: Dataset.from_pandas(dataset)})
    dataset_dict = dataset_dict.map(lambda x: encode_batch(x, tokenizer=tokenizer, max_length=512-pattern['n_token']), batched=True)
    dataset_dict = dataset_dict.map(lambda x: {'input_ids': insert_list(pattern['input_ids'], pattern['text_index'], x['input_ids'], tokenizer.pad_token_id)})
    dataset_dict = dataset_dict.map(lambda x: {'attention_mask': extend_attention_mask(x['attention_mask'], pattern['n_token'])})
    mask_indices = [np.where(np.array(ids) == tokenizer.mask_token_id)[0] for ids in dataset_dict[dict_name]["input_ids"]]
    torch_columns = ["input_ids", "attention_mask", "labels"]
    # @TODO: hat jedes Label einen verbalizer
    for i in range(pattern['n_mask_token']):
        dataset_dict[dict_name] = dataset_dict[dict_name].add_column("mask_indices"+str(i+1), [x[i] for x in mask_indices])
        torch_columns.append("mask_indices"+str(i+1))
    dataset_dict.set_format(type="torch", columns=torch_columns)
    if return_labels:
        return dataset_dict, labels, id2tokenid
    else:
        return dataset_dict, None, None


def evaluate_custom_test_set(config_path, eval_set_path, suffix='eval'):
    logger.info(f'evaluating test_file {eval_set_path}')
    config = read_config(config_path)
    if config['dataset']['out_path']:
        out_path = Path(config['dataset']['out_path'])
    else:
        out_path = Path(config['dataset']['data_path'])
    logger.info('creating dataset for test file')
    for model_id, model_name in config['adapter']['model'].items():
        model_dir = out_path / 'model_id'
        logger.info(f'searching for experiment results in {model_dir}')
        matching_dirs = [file_path.parent for file_path in out_path.rglob('pytorch_adapter.bin')]
        logger.info(f'found {len(matching_dirs)} runs')
        for dir in matching_dirs:
            logger.info(f'evaluating model in {dir}')
            logger.info(f'load model')
            model = AutoAdapterModel.from_pretrained(model_name)
            model.register_custom_head("PEThead", PEThead)
            logger.info(f'load adapter')
            model.load_adapter(str(dir))
            tokenizer = AutoTokenizer.from_pretrained(model_name, additional_special_tokens=["<TEXT>"])
            pattern_name = dir.parts[-3]
            pattern_text = config['dataset']['pattern'][pattern_name]
            pattern = pattern_text.replace("<mask>", tokenizer.mask_token)
            pattern_dict = {
                'pattern' : pattern,
                'n_mask_token' : pattern.count(tokenizer.mask_token),
                'input_ids' : tokenizer(pattern, return_attention_mask=False).input_ids,
            }
            pattern_dict['text_index'] = int(np.where(np.array(pattern_dict['input_ids']) == tokenizer.additional_special_tokens_ids)[0])
            pattern_dict['input_ids'].pop(pattern_dict['text_index'])
            pattern_dict['n_token'] = len(pattern_dict['input_ids'])
            logger.info(f'create dataset')
            dataset_dict, labels, id2tokenid = data_from_csv(
                eval_set_path,
                pattern_dict,
                config['dataset']['verbalizer'],
                tokenizer,
                labels=False,
                split_name='test'
            )
            data = {}
            data['test_dataset'] = dataset_dict
            data['labels'] = labels
            data['id2tokenid'] = id2tokenid
            data['actual'] = dataset_dict["eval"]["labels"].numpy()
            data['input_ids'] = dataset_dict["eval"]["input_ids"].cuda()
            data['attention_mask'] = dataset_dict["eval"]["attention_mask"].cuda()
            for i in range(pattern_dict['n_mask_token']):
                data["mask_indices"+str(i+1)] = dataset_dict["eval"]["mask_indices"+str(i+1)].cuda()
            model.active_adapters = 'myadapter'
            # necessary because loading reads the ints as strings apparently :shrug:
            model.heads.myadapter.config['id2tokenid'] = id2tokenid
            model = model.to('cuda')
            logger.info(f'run evaluate')
            evaluate_model(model, data, dir, suffix)
            gc.collect()
            torch.cuda.empty_cache()


def create_eperiment_data(experiment, pattern_text, verbalizer, tokenizer, test_path):
    logger.debug(f'creating data for {experiment}')
    data = {}
    pattern = pattern_text.replace("<mask>", tokenizer.mask_token)
    pattern = {
    'pattern' : pattern,
    'n_mask_token' : pattern.count(tokenizer.mask_token),
    'input_ids' : tokenizer(pattern, return_attention_mask=False).input_ids,
    }
    pattern['text_index'] = int(np.where(np.array(pattern['input_ids']) == tokenizer.additional_special_tokens_ids)[0])
    pattern['input_ids'].pop(pattern['text_index'])
    pattern['n_token'] = len(pattern['input_ids'])
    ### test
    if test_path == 'as_path':
        test_path = experiment / 'test.csv'
    dataset_test, labels, id2tokenid = data_from_csv(
        test_path, pattern, verbalizer, tokenizer
    )
    ### train
    train_path = experiment / "train.csv"
    dataset, _, _ = data_from_csv(
        train_path, pattern, verbalizer, tokenizer, labels=labels, split_name='train'
    )
    data['test_dataset'] = dataset_test
    data['train_dataset'] = dataset
    data['labels'] = labels
    data['id2tokenid'] = id2tokenid
    data['actual'] = dataset_test["eval"]["labels"].numpy()
    data['input_ids'] = dataset_test["eval"]["input_ids"].cuda()
    data['attention_mask'] = dataset_test["eval"]["attention_mask"].cuda()
    for i in range(pattern['n_mask_token']):
        data["mask_indices"+str(i+1)] = dataset_test["eval"]["mask_indices"+str(i+1)].cuda()
    return data


def evaluate_model(model, data, output_dir, suffix=False):
    if isinstance(output_dir,str):
        output_dir = Path(output_dir)
    logger.debug('put model in eval')
    model.eval()
    preds = [-100] * len(data['input_ids'])
    for i in tqdm(range(len(data['input_ids'])), desc='evaluating model'):
        with torch.no_grad():
            model_args = {key:data[key][i] for key in data.keys() if key in ['input_ids', 'attention_mask'] or 'mask_indices' in key}
            res = model(**model_args)[0]
            preds[i] = np.argmax(res.cpu().detach().numpy(), axis = 1)[0]
    scores = compute_metrics(data['actual'], preds)
    logger.debug('got dem scores')
    with open(output_dir / f"scores{'_'+suffix if suffix else ''}.json", "w") as fp:
        json.dump(scores, fp)
    report = pd.DataFrame(classification_report(data['actual'], preds, output_dict=True)).T
    report.to_csv(f'{str(output_dir)}/classification_report{"_"+suffix if suffix else ""}.csv')
    pd.DataFrame([data['labels'][x] for x in preds]).to_csv(output_dir / f"predictions{'_'+suffix if suffix else ''}.csv", header = False, index = False)


def run_all_experiments(path):
    config = read_config(path)
    dataset_path = Path(config['dataset']['data_path'])
    experiments = find_experiments(dataset_path)
    for experiment in experiments:
        experiment_path = experiment.relative_to(dataset_path)
        for model_name in config['adapter']['model'].keys():
            for pattern_name in config['dataset']['pattern'].keys():
                tokenizer = AutoTokenizer.from_pretrained(config['adapter']['model'][model_name], additional_special_tokens=["<TEXT>"])
                data = create_eperiment_data(experiment,
                                             pattern_text=config['dataset']['pattern'][pattern_name],
                                             verbalizer=config['dataset']['verbalizer'],
                                             tokenizer=tokenizer,
                                             test_path=config['dataset']['test_path'])
                for arch in config['adapter']['arch']:
                    logger.debug(f'running {experiment} on {model_name} with {arch} and {pattern_name} pattern')
                    if arch == "pfeiffer":
                        config_adapter = PfeifferConfig(reduction_factor=config['adapter']['c_rate'])
                    if arch == "pfeifferinv":
                        config_adapter = PfeifferInvConfig(reduction_factor=config['adapter']['c_rate'])
                    if arch == "lora":
                        config_adapter = LoRAConfig(r=config['adapter']['r'], alpha=config['adapter']['alpha'])
                    if arch == "ia3":
                        config_adapter = IA3Config()
                    for run in range(1, config['adapter']['number_of_runs'] + 1):
                        adapter_name = "myadapter"
                        output_dir = Path(config['dataset']['out_path']) / model_name / arch / pattern_name / experiment_path / str(run)
                        logger.debug(f'running {experiment} on {model_name} with {arch} and {pattern_name} pattern | run {run}')
                        training_args = TrainingArguments(
                            seed=int(1895*run),
                            full_determinism=True,
                            learning_rate=config['adapter']['learning_rate'],
                            num_train_epochs=config['adapter']['max_epochs'],
                            logging_strategy="no",
                            evaluation_strategy="no",
                            save_strategy="no",
                            output_dir=output_dir,
                            overwrite_output_dir=True,
                            remove_unused_columns=False,
                            per_device_train_batch_size = config['adapter']['per_device_train_batch_size'],
                        )
                        model = AutoAdapterModel.from_pretrained(config['adapter']['model'][model_name])
                        model.add_adapter(adapter_name, config=config_adapter)
                        model.register_custom_head("PEThead", PEThead)
                        model.add_custom_head(head_type="PEThead", head_name=adapter_name, id2tokenid=data['id2tokenid'])
                        model.train_adapter(adapter_name)
                        logger.debug('start training')
                        trainer = AdapterTrainer(
                            model=model,
                            args=training_args,
                            train_dataset=data['train_dataset']["train"]
                        )
                        training_starttime = time.time()
                        trainer.train()
                        training_endtime = time.time()
                        if arch in ("ia3", "lora"):
                            model.merge_adapter(adapter_name)
                        logger.debug('start evaluation')
                        evaluate_model(model, data, output_dir)
                        evaluate_endtime = time.time()
                        times = {"train": training_endtime-training_starttime, "test": evaluate_endtime-training_endtime}
                        with open(output_dir / "time.json", "w") as fp:
                            json.dump(times, fp)
                        gc.collect()
                        if config['adapter']['save']:
                            model.save_adapter(output_dir, adapter_name)
                        torch.cuda.empty_cache()
                        logger.debug('done with run!')


if __name__ == '__main__':
    cli()
