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
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from transformers import TrainingArguments, AdapterTrainer, PfeifferConfig, PfeifferInvConfig, LoRAConfig, IA3Config
import torch.nn as nn
from transformers.adapters import PredictionHead, CausalLMHead
from datasets import get_dataset_config_names


@click.group()
def cli():
    pass


@cli.command()
@click.argument('path', required=True)
def run(path):
    click.echo('running experiments')
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    run_all_experiments(path)


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
        mask_indices1 = kwargs.get("mask_indices1")
        #mask_indices2 = kwargs.get("mask_indices2")
        
        logits_mask1 = lm_logits[range(lm_logits.shape[0]),mask_indices1,:]
        #logits_mask2 = lm_logits[range(lm_logits.shape[0]),mask_indices2,:]

        id2newid = {i:z for i,z in zip(self.config["id2tokenid_values"], range(len(self.config["id2tokenid_values"])))}
        id2dim = {k: [id2newid[v1] for v1 in v] for k, v in self.config["id2tokenid"].items()}
        verbalizerid = list(id2dim.values())

        logits_mask1 = logits_mask1[:,[k[0] for k in verbalizerid]]
        #logits_mask2 = logits_mask2[:,[k[1] for k in verbalizerid]]

        logits_for_loss = logits_mask1 #+ logits_mask2
        
        if labels is not None:
            loss = loss_fct(logits_for_loss.view(-1, len(self.config["id2tokenid"])), labels.view(-1))
        outputs = (logits_for_loss,) + outputs[1:]
        if loss is not None:
            outputs = (loss,) + outputs
        return outputs
 

def read_config(path, task):
    if type(path) == str:
        path = Path(path)
    with open (path / f'{task}.yml', 'r', encoding='utf8') as cfg:
        config = yaml.safe_load(cfg)
    logger.debug('loaded config')
    config['datapath'] = path / 'data' / task
    if not isinstance(config['adapter']['arch'], list):
        config['adapter']['arch'] = [config['adapter']['arch']]
    if 'per_device_train_batch_size' not in config['adapter']:
        config['adapter']['per_device_train_batch_size'] = 2
    if 'save' not in config['adapter']:
        config['adapter']['save'] = False
    return config


def create_eperiment_data(datapath, pattern, verbalizer, tokenizer):
    logger.debug(f'creating data for {datapath}')
    data = {}
    pattern = pattern.replace("<mask>", tokenizer.mask_token)
    n_mask_token = pattern.count(tokenizer.mask_token)
    pattern_input_ids = tokenizer(pattern, return_attention_mask=False).input_ids
    text_index = int(np.where(np.array(pattern_input_ids) == tokenizer.additional_special_tokens_ids)[0])
    pattern_input_ids.pop(text_index)
    n_token_pattern = len(pattern_input_ids)
    ### test
    text_test = pd.read_csv(datapath / "test.csv", header = None, names = ["text", "label"])
    dataset_test = DatasetDict({'eval': Dataset.from_pandas(text_test)})
    dataset_test = dataset_test.map(lambda x: encode_batch(x, tokenizer=tokenizer, max_length=512-n_token_pattern), batched=True)
    dataset_test = dataset_test.map(lambda x: {'input_ids': insert_list(pattern_input_ids, text_index, x['input_ids'], tokenizer.pad_token_id)})
    dataset_test = dataset_test.map(lambda x: {'attention_mask': extend_attention_mask(x['attention_mask'], n_token_pattern)})
    mask_indices_test = [np.where(np.array(ids) == tokenizer.mask_token_id)[0] for ids in dataset_test["eval"]["input_ids"]]
    # @TODO: in Zukunft ueber Liste von mask_indices - hier aktuell nur ein mask token pro Beobachtung
    dataset_test["eval"] = dataset_test["eval"].add_column("mask_indices1", [x[0] for x in mask_indices_test])
    #dataset_test["eval"] = dataset_test["eval"].add_column("mask_indices2", [x[1] for x in mask_indices_test])
    dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask", "mask_indices1"])#, "mask_indices2"])
    input_ids = dataset_test["eval"]["input_ids"].cuda()
    attention_mask = dataset_test["eval"]["attention_mask"].cuda()
    mask_indices1 = dataset_test["eval"]["mask_indices1"].cuda()
    #mask_indices2 = dataset_test["eval"]["mask_indices2"].cuda()
    ### train
    text_train = pd.read_csv(datapath / "train.csv", header = None, names = ["text", "label"])
    labels = list(text_train.label.unique())
    labels.sort()
    id2tokenid = {idx: tokenizer(verbalizer[label], add_special_tokens=False, return_attention_mask=False).input_ids[0:n_mask_token] for idx, label in enumerate(labels)}
    text_train['labels'] = [labels.index(x) for x in text_train['label']]
    dataset = DatasetDict({'train': Dataset.from_pandas(text_train)})
    dataset = dataset.map(lambda x: encode_batch(x, tokenizer=tokenizer, max_length=512-n_token_pattern), batched=True)
    dataset = dataset.map(lambda x: {'input_ids': insert_list(pattern_input_ids, text_index, x['input_ids'], tokenizer.pad_token_id)})
    dataset = dataset.map(lambda x: {'attention_mask': extend_attention_mask(x['attention_mask'], n_token_pattern)})
    mask_indices = [np.where(np.array(ids) == tokenizer.mask_token_id)[0] for ids in dataset["train"]["input_ids"]]
    dataset["train"] = dataset["train"].add_column("mask_indices1", [x[0] for x in mask_indices])
    #dataset["train"] = dataset["train"].add_column("mask_indices2", [x[1] for x in mask_indices])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "mask_indices1"])#, "mask_indices2"])
    data['test_dataset'] = dataset_test
    data['train_dataset'] = dataset
    data['input_ids'] = input_ids
    data['attention_mask'] = attention_mask
    data['mask_indices1'] = mask_indices1
    #data['mask_indices2'] = mask_indices2
    data['labels'] = labels
    data['id2tokenid'] = id2tokenid
    return data


def create_prediction_file(model, data, output_dir):
    model.eval()
    preds = [-100] * len(data['input_ids'])
    result_df = pd.DataFrame(columns=data['labels'])
    for i in range(len(data['input_ids'])):
        with torch.no_grad():
            res = model(input_ids = data['input_ids'][i], attention_mask = data['attention_mask'][i],
                        mask_indices1 = data['mask_indices1'][i])[0]#, mask_indices2 = data['mask_indices2'][i])[0]
        res = res.cpu().detach().numpy()
        result_df = pd.concat([result_df, pd.DataFrame(res, columns=data['labels'])])
        preds[i] = np.argmax(res, axis = 1)[0]
    pd.DataFrame({"ID": list(range(50,len(preds)+50)),
                  "Label": [data['labels'][x] for x in preds]}).to_csv(output_dir / "predictions.csv", index=False)
    result_df.to_csv(output_dir / "likelihoods.csv", index=False)


def run_all_experiments(path):
    tasks = get_dataset_config_names("ought/raft")
    tasks.sort()
    tasks = ["tweet_eval_hate", "twitter_complaints", "ade_corpus_v2", "banking_77"]
    for task in tasks:
        config = read_config(path, task)
        tokenizer = AutoTokenizer.from_pretrained(config['adapter']['model'], additional_special_tokens=["<TEXT>"])
        data = create_eperiment_data(
            datapath=config['datapath'],
            pattern=config['pvp']['pattern'],
            verbalizer=config['pvp']['verbalizer'],
            tokenizer=tokenizer)
        for arch in config['adapter']['arch']:
            logger.debug(f"running {task} on {config['adapter']['model']} with {arch}")
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
                output_dir = Path("raft") / task / arch / str(run)
                logger.debug(f"running {task} on {config['adapter']['model']} with {arch} | run {run}")
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
                model = AutoAdapterModel.from_pretrained(config['adapter']['model'])
                model.add_adapter(adapter_name, config=config_adapter)
                model.register_custom_head("PEThead", PEThead)
                model.add_custom_head(head_type="PEThead", head_name=adapter_name, id2tokenid=data['id2tokenid'])
                model.train_adapter(adapter_name)
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
                create_prediction_file(model, data, output_dir)
                prediction_endtime = time.time()
                times = {"train": training_endtime-training_starttime, "prediction": prediction_endtime-training_endtime}
                with open(output_dir / "time.json", "w") as fp:
                    json.dump(times, fp)
                gc.collect()
                if config['adapter']['save']:
                    model.save_adapter(output_dir, adapter_name)
                torch.cuda.empty_cache()
                    

if __name__ == '__main__':
    cli()
