#!/bin/bash
#SBATCH -p gpu-preempt  # Partition
#SBATCH --constraint m40  # GPU type
#SBATCH -G 1  # Number of GPUs
#SBATCH -c 1  # Number of CPU cores
#SBATCH -t 05:00:00  # Runtime in D-HH:MM:SS
#SBATCH --mem=30GB  # Requested Memory
#SBATCH -o logs/gpt2-xl/wikitext-2/perplexity-W8A8.out  # File to which STDOUT will be written

export WL=/work/pi_dhruveshpate_umass_edu/debanjanmond_umass_edu
export HF_HOME=$WL
module load miniconda/22.11.1-1
conda activate zeroquant
 python decoder-perplexity.py --dataset  wikitext --dataset-config wikitext-2-raw-v1 --model gpt2-xl --tokenizer gpt2-xl  --batch-size 4 \
--quant-config quant_configs/gpt2_xl_config_W8A8.json