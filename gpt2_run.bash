#!/bin/bash

GLUE=("cola" "mrpc" "mnli" "sst2" "qqp" "qnli" "rte" "stsb" "wnli")
PRECISIONS=("W4_8A8" "W4_8A16" "W8A8")
MODEL=("gpt2-finetuned")

# Install Git LFS
git lfs install

# Loop through each dataset and clone
for task in "${GLUE[@]}"
do
    for model_name in "${MODEL[@]}"
    do
        rm -rf ./"$model_name-$task"
        git clone https://huggingface.co/PavanNeerudu/"$model_name-$task"
        
        for precision in "${PRECISIONS[@]}"
        do
            echo "Currently Running for ${model_name}-${task}-${precision}"
            echo "--------------------------- START ----------------------------------"
            # Check if bert-main.py exists and execute it
            if [ -f "gpt2-main.py" ]; then
                MODEL_CONFIG=$(echo "$model_name" | sed 's/-/_/g')
                python gpt2-main.py --model-name $model_name --task-name $task --precision $precision --quant-config quant_configs/${MODEL_CONFIG}_config_${precision}.json
            else
                echo "gpt2-main.py not found!"
            fi
            echo "--------------------------- FINISH ----------------------------------"
        done
        rm -rf ./"$model_name-$task"
    done
done