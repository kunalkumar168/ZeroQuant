#!/bin/bash

GLUE=("cola" "mrpc" "mnli" "stsb" "sst2" "qqp" "qnli" "rte" "wnli")
PRECISIONS=("W4_8A8" "W4_8A16" "W8A8")
MODEL_NAMES=("bert-base-uncased" "bert-large-uncased")

# Function to run Python script for each task
for task in "${GLUE[@]}"
do  
    for model_name in "${MODEL_NAMES[@]}"
    do
        for precision in "${PRECISIONS[@]}"
        do
            echo "Currently Running for ${model_name}-${task}-${precision}"
            echo "--------------------------- START ----------------------------------"
            # Check if bert-main.py exists and execute it
            if [ -f "bert-main.py" ]; then
                MODEL_CONFIG=$(echo "$model_name" | sed 's/-/_/g')
                python bert-main.py --model-name $model_name --task-name $task --precision $precision --quant-config quant_configs/${MODEL_CONFIG}_config_${precision}.json
            else
                echo "bert-main.py not found!"
            fi
            echo "--------------------------- FINISH ----------------------------------"
        done
    done
done
