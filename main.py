from transformers import AutoTokenizer, BertForSequenceClassification, AutoConfig, PretrainedConfig, default_data_collator, AdamW
from datasets import load_dataset, load_metric
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from torch.cuda.amp import autocast
import json
from compress import compress

task_name = 'mnli'
raw_datasets = load_dataset("glue", task_name)
raw_datasets

model_name = 'yoshitomo-matsubara/bert-base-uncased-mnli'

labels = raw_datasets['train'].features['label'].names

num_labels = len(labels)
config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, finetuning_task='mnli')

model = BertForSequenceClassification.from_pretrained(model_name, config=config, from_tf=False)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def get_model_size(model):
    size = sum([p.numel() * p.element_size() for p in model.parameters() if p.requires_grad])

    size_in_millions = size / 10 ** 6
    return f'{size_in_millions:.1f} GB'

print('Size of the original BERT model', get_model_size(model))

sentence1_key, sentence2_key = 'premise', 'hypothesis'

model.config.label2id = { l: i for i, l in enumerate(labels)}
model.config.id2label = { i: l for i, l in enumerate(labels) }

def process(examples):
    texts = (examples[sentence1_key] if not sentence2_key else (examples[sentence1_key], examples[sentence2_key]))
    data = tokenizer(*texts, padding="max_length", max_length=128, truncation=True)
    data['labels'] = examples['label']
    return data

processed_dataset = raw_datasets.map(
    process,
    batched=True,
    remove_columns=raw_datasets['train'].column_names
)

train_dataset = processed_dataset['train']
valid_dataset = processed_dataset['validation_matched']

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(
    train_dataset,
    collate_fn=default_data_collator,
    sampler=train_sampler,
    batch_size=16
)

valid_sampler = SequentialSampler(valid_dataset)
valid_dataloader = DataLoader(valid_dataset,
                    collate_fn=default_data_collator,
                    sampler=valid_sampler,
                    batch_size=256
)

@torch.inference_mode()
def evaluate(model, valid_dataloader):
    model.eval()
    with autocast():
        metric = load_metric('glue', 'mnli', trust_remote_code=True)
        for batch in tqdm(valid_dataloader):
            batch = { k:v.to(device) for k, v in batch.items() }
            logits = model(**batch).logits
            predictions = logits.argmax(dim=-1)
            metric.add_batch(predictions=predictions, references=batch['labels'])
    return metric.compute()

print('Accuracy before quantization', evaluate(model, valid_dataloader))


quant_config = None
with open('quant_configs/bert_config.json', 'r') as f:
    quant_config = json.load(f)

qunatized_model = compress(model, quant_config)

print('Accuracy after quantization', evaluate(qunatized_model, valid_dataloader))

print('Size of the quantized BERT model', get_model_size(qunatized_model))
