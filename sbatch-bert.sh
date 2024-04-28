#!/bin/bash
#SBATCH -p gpu-preempt  # Partition
#SBATCH --constraint m40  # GPU type
#SBATCH -G 1  # Number of GPUs
#SBATCH -c 1  # Number of CPU cores
#SBATCH -t 05:00:00  # Runtime in D-HH:MM:SS
#SBATCH --mem=20GB  # Requested Memory
#SBATCH -o logs/bert-base-W4_8A16.out  # File to which STDOUT will be written

export WL=/work/pi_dhruveshpate_umass_edu/debanjanmond_umass_edu
export HF_HOME=$WL
module load miniconda/22.11.1-1
conda activate zeroquant
python bert-main.py --quant-config quant_configs/bert_base_config_W4_8A16.json