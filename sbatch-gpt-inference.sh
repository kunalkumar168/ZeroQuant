#!/bin/bash
#SBATCH -p gpu-preempt  # Partition
#SBATCH --constraint m40  # GPU type
#SBATCH -G 1  # Number of GPUs
#SBATCH -c 1  # Number of CPU cores
#SBATCH -t 05:00:00  # Runtime in D-HH:MM:SS
#SBATCH --mem=30GB  # Requested Memory
#SBATCH -o logs/gpt2-medium/lambada/perplexity-W8A8.out  # File to which STDOUT will be written

export WL=/work/pi_dhruveshpate_umass_edu/debanjanmond_umass_edu
export HF_HOME=$WL
module load miniconda/22.11.1-1
conda activate zeroquant
 python decoder-perplexity.py --dataset  cimec/lambada --model gpt2-medium --tokenizer gpt2-medium  --batch-size 4 \
--quant-config quant_configs/gpt2_medium_config_W8A8.json