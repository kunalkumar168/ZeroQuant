from transformers import AutoTokenizer, BertForSequenceClassification, AutoConfig, PretrainedConfig, default_data_collator, AdamW
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

ACC_TASKS = ["mnli", "mrpc", "sst2", "qqp", "qnli", "rte"]

TASKS_TO_KEYS = {
    "stsb": ("sentence1", "sentence2"),
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "wnli": ("sentence1", "sentence2")
}

def get_memory_footprint(model):
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    return mem/(1e9)

def main(model, quant_config, valid_dataloader, task_name, device, is_regression=False):

    print('Processing task:', task_name)
    old_model_size = get_memory_footprint(model)
    print('Size before Quantization', old_model_size)
    print('Accuracy before quantization', evaluate(model, valid_dataloader, task_name, device, is_regression))

    # qunatized_model = compress(model, quant_config)

    # print('Accuracy after quantization', evaluate(qunatized_model, valid_dataloader, task_name, device))

    fixed_quantized_model = fix_compression(model, quant_config)
    quant_model_size = get_memory_footprint(fixed_quantized_model)

    print('Size after Quantization', quant_model_size)
    print('Compression ratio:', old_model_size / quant_model_size)

    print('Accuracy after quantization', evaluate(fixed_quantized_model, valid_dataloader, task_name, device, is_regression))

    print('-----' * 20)
    print('\n' * 20)

@torch.inference_mode()
def evaluate(model, valid_dataloader, task_name, device, is_regression=False):
    model.eval()
    metric = load_metric('glue', task_name, trust_remote_code=True)
    time_required = []
    for batch in valid_dataloader:
        batch = { k:v.to(device) for k, v in batch.items() }
        start_time = time.time()
        logits = model(**batch).logits
        time_required.append(time.time() - start_time)
        predictions = logits.argmax(dim=-1) if not is_regression else logits.squeeze()
        metric.add_batch(predictions=predictions, references=batch['labels'])
    print('Average time taken for batch:', np.mean(time_required))
    eval_metrics = metric.compute()
    eval_metrics['average'] = np.mean(list(eval_metrics.values()))
    return eval_metrics

def process_task(task_name):
    task_name = task_name
    raw_datasets = load_dataset("glue", task_name)
    model_name = f'yoshitomo-matsubara/bert-large-uncased-{task_name}'

    is_regression = task_name == "stsb"
    if not is_regression:
        labels = raw_datasets["train"].features["label"].names
        num_labels = len(labels)
    else:
        labels = None
        num_labels = 1

    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, finetuning_task=task_name)

    model = BertForSequenceClassification.from_pretrained(model_name, config=config, from_tf=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).half()
    sentence1_key, sentence2_key = TASKS_TO_KEYS[task_name]
    label_to_id = None

    if PretrainedConfig(num_labels=num_labels).label2id != model.config.label2id and not is_regression:
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        label_to_id = {i: label_name_to_id[labels[i]] for i in range(num_labels)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif task_name is not None and not is_regression:
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

    valid_dataset = processed_dataset["validation_matched" if task_name == "mnli" else "validation"]


    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset,
                        collate_fn=default_data_collator,
                        sampler=valid_sampler,
                        batch_size=256
    )

    main(model, quant_config, valid_dataloader, task_name, device, is_regression)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT quantization')
    parser.add_argument('--task-name', type=str, default=None, help='GLUE Task name')
    parser.add_argument('--quant-config', type=str, default='bert_config.json', help='Quantization config file')
    args = parser.parse_args()


    quant_config = None
    with open(args.quant_config, 'r') as f:
        quant_config = json.load(f)

    print('Using quantization config')
    print(quant_config)


    task_name = args.task_name
    if not args.task_name:
        task_name = list(TASKS_TO_KEYS.keys())

    if isinstance(task_name, list):
        for task in task_name:
            process_task(task)
    else:
        process_task(task_name)