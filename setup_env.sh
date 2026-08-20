#!/usr/bin/env bash
set -e

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name credit-risk-env --display-name "Python (Credit Risk)"

echo "Environment ready."
echo "Activate later with: source .venv/bin/activate"
