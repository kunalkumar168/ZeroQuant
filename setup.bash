#!/bin/bash

# Install Python venv
sudo apt update
sudo apt install -y python3-venv

# Create and activate virtual environment
python3 -m venv zeroquant
source zeroquant/bin/activate

# Install required Python packages
pip install -r requirements.txt