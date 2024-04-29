#!/bin/bash

GLUE=("cola" "mrpc" "mnli" "sst2" "qqp" "qnli" "rte" "stsb" "wnli")
MODEL_NAME="roberta-large" #roberta-base

# Install Git LFS
git lfs install

# Loop through each dataset and clone
for task in "${GLUE[@]}"
do
    rm -rf ./"$MODEL_NAME-$task"
    if [ "$MODEL_NAME" = "roberta-base" ]; then
        if [ "$task" = "mrpc" ]; then
            git clone https://huggingface.co/Intel/"$MODEL_NAME-$task"
        elif [ "$task" = "wnli" ]; then
            git clone https://huggingface.co/JeremiahZ/"$MODEL_NAME-$task"
        else
            git clone https://huggingface.co/WillHeld/"$MODEL_NAME-$task"
        fi
    elif [ "$MODEL_NAME" = "roberta-large" ]; then
        if [ "$task" = "wnli" ]; then
            continue
        else
            git clone https://huggingface.co/howey/"$MODEL_NAME-$task"
        fi
    fi
    
    # Check if roberta-main.py exists and execute it
    if [ -f "roberta-main.py" ]; then
        MODEL_CONFIG=$(echo "$MODEL_NAME" | sed 's/-/_/g')
        python roberta-main.py --model-name $MODEL_NAME --task-name $task --quant-config quant_configs/${MODEL_CONFIG}_config.json
    else
        echo "roberta-main.py not found!"
    fi
    
    rm -rf ./"$MODEL_NAME-$task"
done