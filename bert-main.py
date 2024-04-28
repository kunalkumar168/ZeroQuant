import os
from transformers import AutoTokenizer, BertForSequenceClassification, AutoConfig, PretrainedConfig, default_data_collator, AdamW
from datasets import load_dataset, load_metric
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from torch.cuda.amp import autocast
import json
from compress import compress, fix_compression
from utils import log_profiling_metrics, get_logger, write_quant_results_to_file
import argparse
import time
from copy import deepcopy
import numpy as np
import logging

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
    
def main(model, quant_config, valid_dataloader, task_name, device, profiling_path, logger):
    model_name = quant_config['model']['name']
    logger.info('MODEL NAME: %s', model_name)
    logger.info('TASK NAME: %s', task_name)
    logger.info('DEVICE: %s', device)

    logger.info('Evaluating model before quantization...')
    model_size = get_memory_footprint(model)
    logger.info('Size before Quantization (GB): %s', model_size)
    accuracy = evaluate(model, valid_dataloader, task_name, device, logger, profiling_path, model_name, is_quantized=False)
    logger.info('Accuracy before quantization: %s', accuracy)

    logger.info('Quantizing model...')
    fixed_quantized_model = fix_compression(model, quant_config)
    quantized_size = get_memory_footprint(fixed_quantized_model)
    logger.info('Size after Quantization (GB): %s', quantized_size)
    quantized_accuracy = evaluate(fixed_quantized_model, valid_dataloader, task_name, device, logger, profiling_path, model_name, is_quantized=True)
    logger.info('Accuracy after quantization: %s', quantized_accuracy)
    
    write_quant_results_to_file(model_size, accuracy, quantized_size, quantized_accuracy, task_name, logger)

@torch.inference_mode()
def evaluate(model, valid_dataloader, task_name, device, logger, profiling_path, model_name, is_quantized=False):
    model.eval()
    model.to(device)
    metric = load_metric('glue', task_name, trust_remote_code=True)
    time_required = []

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for batch in tqdm(valid_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            start_time = time.time()
            with record_function("model_inference"):
                logits = model(**batch).logits
            time_required.append(time.time() - start_time)
            predictions = logits.argmax(dim=-1)
            metric.add_batch(predictions=predictions, references=batch['labels'])

    log_profiling_metrics(prof, model_name, task_name, logger, file_path=profiling_path, is_quantized=is_quantized)
    logger.info('Average time taken for batch: %s', np.mean(time_required))

    return metric.compute()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT quantization')
    parser.add_argument('--model-name', type=str, default='bert_base', help='Model name')
    parser.add_argument('--task-name', type=str, default='mnli', help='Task name')
    parser.add_argument('--quant-config', type=str, default='bert_config.json', help='Quantization config file')
    parser.add_argument('--profiling-path', type=str, default='profiling_results', help='Directory to save profiling results')
    parser.add_argument('--logging-file-path', type=str, default='logs.log', help='Directory to save logs')
    args = parser.parse_args()


    quant_config = None
    with open(args.quant_config, 'r') as f:
        quant_config = json.load(f)

    logger = get_logger(args.model_name, args.task_name, args.logging_file_path)

    task_name = args.task_name
    raw_datasets = load_dataset("glue", task_name)
    model_name = f'yoshitomo-matsubara/bert-base-uncased-{args.task_name}'

    is_regression = args.task_name == "stsb"
    if not is_regression:
        labels = raw_datasets["train"].features["label"].names
        num_labels = len(labels)
    else:
        labels = None
        num_labels = 1

    num_labels = len(labels)
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, finetuning_task=args.task_name)

    model = BertForSequenceClassification.from_pretrained(model_name, config=config, from_tf=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sentence1_key, sentence2_key = TASKS_TO_KEYS[args.task_name]
    label_to_id = None

    if PretrainedConfig(num_labels=num_labels).label2id != model.config.label2id and not is_regression:
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        label_to_id = {i: label_name_to_id[labels[i]] for i in range(num_labels)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    else:
        model.config.label2id = { l: i for i, l in enumerate(labels)}
        model.config.id2label = { i: l for i, l in enumerate(labels) }

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

    main(model, quant_config, valid_dataloader, args.task_name, device, args.profiling_path, logger)
    logger.info('*****DONE with the current task!*****')