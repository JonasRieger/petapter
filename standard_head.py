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

def read_config(path):
    if type(path) == str:
        path = Path(path)
    with open (path / 'config_standard_head.yml', 'r', encoding='utf8') as cfg:
        config = yaml.safe_load(cfg)
    logger.debug('loaded config')
    config['dataset']['train_path'] = path / ".." / 'data' / 'train' / config['dataset']['data']
    config['dataset']['test_path'] = path / ".." / 'data' / 'test' / config['dataset']['data']
    if not isinstance(config['adapter']['arch'], list):
        config['adapter']['arch'] = [config['adapter']['arch']]
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


def create_eperiment_data(experiment, tokenizer, test_path):
    logger.debug(f'creating data for {experiment}')
    data = {}
    ### test
    text_test = pd.read_csv(test_path / "test.csv", header = None, names = ["text", "label"])
    labels = list(text_test.label.unique())
    labels.sort()
    id2label = {idx:label for idx,label in enumerate(labels)}
    text_test['labels'] = [labels.index(x) for x in text_test['label']]
    dataset_test = DatasetDict({'eval': Dataset.from_pandas(text_test)})
    dataset_test = dataset_test.map(lambda x: encode_batch(x, tokenizer=tokenizer, max_length=512), batched=True)
    dataset_test = dataset_test.map(lambda x: {'input_ids': insert_list([], 0, x['input_ids'], tokenizer.pad_token_id)})
    dataset_test = dataset_test.map(lambda x: {'attention_mask': extend_attention_mask(x['attention_mask'], 0)})
    torch_columns = ["input_ids", "attention_mask", "labels"]
    dataset_test.set_format(type="torch", columns=torch_columns)
    ### train
    text_train = pd.read_csv(experiment / "train.csv", header = None, names = ["text", "label"])
    text_train['labels'] = [labels.index(x) for x in text_train['label']]
    dataset = DatasetDict({'train': Dataset.from_pandas(text_train)})
    dataset = dataset.map(lambda x: encode_batch(x, tokenizer=tokenizer, max_length=512), batched=True)
    dataset = dataset.map(lambda x: {'input_ids': insert_list([], 0, x['input_ids'], tokenizer.pad_token_id)})
    dataset = dataset.map(lambda x: {'attention_mask': extend_attention_mask(x['attention_mask'], 0)})
    dataset.set_format(type="torch", columns=torch_columns)

    data['train_dataset'] = dataset
    data['labels'] = labels
    data['id2label'] = id2label
    data['actual'] = dataset_test["eval"]["labels"].numpy()
    data['input_ids'] = dataset_test["eval"]["input_ids"].cuda()
    data['attention_mask'] = dataset_test["eval"]["attention_mask"].cuda()
    return data


def evaluate_model(model, data, output_dir):
    model.eval()
    preds = [-100] * len(data['input_ids'])
    for i in range(len(data['input_ids'])):
        with torch.no_grad():
            model_args = {key:data[key][i] for key in data.keys() if key in ['input_ids', 'attention_mask'] or 'mask_indices' in key}
            res = model(**model_args)[0]
            preds[i] = np.argmax(res.cpu().detach().numpy(), axis = 1)[0]
    scores = compute_metrics(data['actual'], preds)
    with open(output_dir / "scores.json", "w") as fp:
        json.dump(scores, fp)
    pd.DataFrame([data['labels'][x] for x in preds]).to_csv(output_dir / "predictions.csv", header = False, index = False)


def run_all_experiments(path):
    config = read_config(path)
    dataset_path = Path(config['dataset']['train_path'])
    experiments = find_experiments(dataset_path)
    for experiment in experiments:
        experiment_path = experiment.relative_to(dataset_path)
        for model_name in config['adapter']['model'].keys():
            tokenizer = AutoTokenizer.from_pretrained(config['adapter']['model'][model_name])
            data = create_eperiment_data(experiment,
                                         tokenizer=tokenizer,
                                         test_path=config['dataset']['test_path'])
            for arch in config['adapter']['arch']:
                logger.debug(f'running {experiment} on {model_name} with {arch}')
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
                    output_dir = Path(config['dataset']['data']) / model_name / arch / "standard_head" / experiment_path / str(run)
                    logger.debug(f'running {experiment} on {model_name} with {arch} | run {run}')
                    training_args = TrainingArguments(
                        seed=int(1895 * run),
                        full_determinism=True,
                        learning_rate=config['adapter']['learning_rate'],
                        num_train_epochs=config['adapter']['max_epochs'],
                        logging_strategy="no",
                        evaluation_strategy="no",
                        save_strategy="no",
                        output_dir=output_dir,
                        overwrite_output_dir=True,
                        remove_unused_columns=False,
                        per_device_train_batch_size=config['adapter']['per_device_train_batch_size'],
                    )
                    model = AutoAdapterModel.from_pretrained(config['adapter']['model'][model_name])
                    model.add_adapter(adapter_name, config=config_adapter)
                    model.add_classification_head(adapter_name, num_labels=len(data['id2label']), id2label=data['id2label'])
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
                    evaluate_model(model, data, output_dir)
                    evaluate_endtime = time.time()
                    times = {"train": training_endtime - training_starttime,
                             "test": evaluate_endtime - training_endtime}
                    with open(output_dir / "time.json", "w") as fp:
                        json.dump(times, fp)
                    gc.collect()
                    if config['adapter']['save']:
                        model.save_adapter(output_dir, adapter_name)
                    torch.cuda.empty_cache()
                    

if __name__ == '__main__':
    cli()
