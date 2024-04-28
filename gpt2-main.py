from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, PretrainedConfig, default_data_collator, AdamW
from datasets import load_dataset, load_metric
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from torch.cuda.amp import autocast
import json
from compress import compress, fix_compression
import argparse
import time
from copy import deepcopy
import numpy as np
import os
import math

ACC_TASKS = ["mnli", "mrpc", "sst2", "qqp", "qnli", "rte"]

TASKS_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def get_memory_footprint(model):
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    return mem/(1e9)

def main(model_name, model, quant_config, valid_dataloader, task_name, device, valid_dataset, is_regression):
    model_size = get_memory_footprint(model)
    print('Size before Quantization (GB)', model_size)
    accuracy = evaluate(model, valid_dataloader, task_name, device, is_regression)
    print('Accuracy before quantization', accuracy)

    fixed_quantized_model = fix_compression(model, quant_config)
    quantized_size = get_memory_footprint(fixed_quantized_model)
    print('Size after Quantization (GB)', quantized_size)
    quantized_accuracy = evaluate(model, valid_dataloader, task_name, device, is_regression)
    print('Accuracy after quantization', quantized_accuracy)
    
    currpath = os.path.abspath(os.curdir)
    with open(os.path.join(f'{currpath}/results', f'{model_name}-{task_name}.txt'), 'w') as f:
        f.write(f"Size before Quantization (GB) : {model_size}\n")
        f.write(f"Accuracy before quantization : {accuracy}\n")
        f.write(f"-----------------------------------\n")
        f.write(f"Size after Quantization (GB): {quantized_size}\n")
        f.write(f"Accuracy after quantization : {quantized_accuracy}")
        f.close() 

@torch.inference_mode()
def evaluate(model, valid_dataloader, task_name, device, is_regression=False):
    model.eval()
    model.to(device)
    if task_name is not None:
        metric = load_metric("glue", task_name, trust_remote_code=True)
    else:
        metric = load_metric("accuracy")
    time_required = []
    for batch in tqdm(valid_dataloader):
        batch = { k:v.to(device) for k, v in batch.items() }
        start_time = time.time()
        logits = model(**batch).logits
        time_required.append(time.time() - start_time)
        predictions = logits.argmax(dim=-1) if not is_regression else logits.squeeze()
        metric.add_batch(predictions=predictions, references=batch['labels'])
    print('Average time taken for batch:', np.mean(time_required))
    return metric.compute()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RoBERTa quantization')
    parser.add_argument('--model-name', type=str, default='roberta-base', help='Model name')
    parser.add_argument('--task-name', type=str, default='mnli', help='Task name')
    parser.add_argument('--quant-config', type=str, default='bert_config.json', help='Quantization config file')
    args = parser.parse_args()


    quant_config = None
    with open(args.quant_config, 'r') as f:
        quant_config = json.load(f)


    task_name = args.task_name
    model_name = args.model_name
    currpath = os.path.abspath(os.curdir)
    model_path = f'{currpath}/{model_name}-{task_name}'
    raw_datasets = load_dataset("glue", task_name)

    is_regression = args.task_name == "stsb"
    num_labels = 0
    if not is_regression:
        labels = raw_datasets["train"].features["label"].names
        num_labels = len(labels)
    else:
        labels = None
        num_labels = 1

    config = AutoConfig.from_pretrained(model_path, num_labels=num_labels, finetuning_task=args.task_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, from_tf=False, ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding='max_length', truncation=True)
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sentence1_key, sentence2_key = TASKS_TO_KEYS[args.task_name]
    label_to_id = None

    if PretrainedConfig(num_labels=num_labels).label2id != model.config.label2id and not is_regression:
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        label_to_id = {i: label_name_to_id[labels[i]] for i in range(num_labels)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = { l: i for i, l in enumerate(labels)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    def process(examples):
        texts = ((examples[sentence1_key],) if not sentence2_key else (examples[sentence1_key], examples[sentence2_key]))
        data = tokenizer(*texts, padding='max_length', max_length=128, truncation=True)
        if 'label' in examples:
            data['labels'] = examples['label']
        return data

    processed_dataset = raw_datasets.map(
        process,
        batched=True,
        remove_columns=raw_datasets['train'].column_names,
        desc='Running tokenizer on dataset'
    )

    valid_dataset = processed_dataset["validation_matched" if args.task_name == "mnli" else "validation"]

    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset,
                        collate_fn=default_data_collator,
                        sampler=valid_sampler,
                        batch_size=256
    )

    main(model_name, model, quant_config, valid_dataloader, args.task_name, device, valid_dataset, is_regression)