#!/bin/bash

GLUE=("cola" "mrpc" "mnli" "sst2" "qqp" "qnli" "rte" "stsb" "wnli")

# Function to run Python script for each task
for task in "${GLUE[@]}"
do  
    python bert-main.py --model-name bert-base --task-name $task --quant-config quant_configs/bert_config.json
done
