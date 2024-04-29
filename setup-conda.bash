#!/bin/bash


# Create and activate virtual environment
PATH="/work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/"
ENV="zq_env"
conda create --prefix "$PATH/$ENV" --name zquant python=3.8 -y
source activate zquant

# Install required Python packages
# pip install -r requirements.txt