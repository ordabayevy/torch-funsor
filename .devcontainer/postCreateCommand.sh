#!/usr/bin/env bash

# Set up execution environment
# conda init bash
# source ~/.bashrc
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate dev
# python -m ipykernel install --name dev
python -m ipykernel install --user --name dev

pip install -e .

# Add local bin to PATH permanently
echo 'export PATH="/home/vscode/.local/bin:$PATH"' >> ~/.bashrc
export PATH="/home/vscode/.local/bin:$PATH"