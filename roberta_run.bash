#!/bin/bash

#GLUE=("cola" "mrpc" "mnli" "sst2" "qqp" "qnli" "rte" "stsb" "wnli")
GLUE=("stsb" "wnli")

# Install Git LFS
git lfs install

# Loop through each dataset and clone
for task in "${GLUE[@]}"
do
    rm -rf ./roberta-base-$task
    if [ "$task" = "mrpc" ]; then
        git clone https://huggingface.co/Intel/roberta-base-$task
    elif [ "$task" = "wnli" ]; then
        git clone https://huggingface.co/JeremiahZ/roberta-base-$task
    else
        git clone https://huggingface.co/WillHeld/roberta-base-$task
    fi
    
    python roberta-main.py --model-name roberta-base --task-name $task --quant-config quant_configs/roberta_config.json
    rm -rf ./roberta-base-$task
done