import argparse
import torch
from datasets import load_dataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, default_data_collator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import os
import sys
from tqdm import tqdm
from compress import fix_compression
import json

def evaluate_perplexity(model, eval_dataloader, device):
    model.eval()
    total_loss = []
    # total_loss = torch.tensor(0.0, device=device)
    total_length = torch.tensor(0.0, device=device)
    for batch in tqdm(eval_dataloader):
        # batch = tuple(t.to(device) for t in batch)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        # total_loss += loss * batch['input_ids'].size(1)
        # total_length += batch['input_ids'].size(1)
        total_loss.append(loss.cpu().item())
    # total_loss = total_loss[: len(eval_dataset)]

     
    try:
        # average_loss = total_loss / total_length
        perplexity = math.exp(np.mean(total_loss))
    except OverflowError:
        perplexity = float("inf")
    return perplexity

def get_model_and_tokenizer(args, device_id):
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config).to(device_id)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def main(args, quant_config):
    device = torch.device("cuda:{}".format(0))
    
    raw_datasets = load_dataset(args.dataset, args.dataset_config, split='test')

    model, tokenizer = get_model_and_tokenizer(args, device)
    print('Model loaded')

    text_column_name = "text"

    def process_text(examples):
        return tokenizer(examples[text_column_name])
    
    tokenized_datasets = raw_datasets.map(
        process_text,
        batched=True,
        remove_columns=raw_datasets.column_names,
        desc="Running tokenizer on dataset"
    )

    block_size = min(1024, tokenizer.model_max_length)

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    eval_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}"
    )

    eval_sampler = SequentialSampler(eval_dataset)

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        sampler=eval_sampler,
        batch_size=args.batch_size
    )
    
    print('dataset loaded')

    model = model.half()
    old_model_size = model.get_memory_footprint()
    print('Size before Quantization', old_model_size)
    print('Perplexity before quantization',  evaluate_perplexity(model, eval_dataloader, device))

    # qunatized_model = compress(model, quant_config)

    # print('Accuracy after quantization', evaluate(qunatized_model, valid_dataloader, task_name, device))

    fixed_quantized_model = fix_compression(model, quant_config)
    quant_model_size = fixed_quantized_model.get_memory_footprint()

    print('Size after Quantization', quant_model_size)
    print('Compression ratio:', old_model_size / quant_model_size)

    print('Perplexity after quantization', evaluate_perplexity(fixed_quantized_model, eval_dataloader, device))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decoder model quantization')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--dataset-config', type=str, default=None, help='Model name')
    parser.add_argument('--model', type=str, default='gpt2-large', help='Model name')
    parser.add_argument('--tokenizer', type=str, default='gpt2-large', help='Tokenizer')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--quant-config', type=str, default=None, help='Quant Config')

    args = parser.parse_args()

    quant_config = None
    with open(args.quant_config, 'r') as f:
        quant_config = json.load(f)

    print('Using quantization config')
    print(quant_config)

    main(args, quant_config)



