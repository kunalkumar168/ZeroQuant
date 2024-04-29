#!/bin/bash

GLUE=("cola" "mrpc" "mnli" "sst2" "qqp" "qnli" "rte" "stsb" "wnli")
MODEL="gpt2-finetuned"

# Install Git LFS
git lfs install

# Loop through each dataset and clone
for task in "${GLUE[@]}"
do
    rm -rf ./"$MODEL-$task"
    git clone https://huggingface.co/PavanNeerudu/"$MODEL-$task"
    python gpt2-main.py --model-name $MODEL --task-name $task --quant-config quant_configs/roberta_base_config.json
    rm -rf ./"$MODEL-$task"
done