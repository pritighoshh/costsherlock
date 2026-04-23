#!/usr/bin/env bash
# CostSherlock environment setup

set -euo pipefail

python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Activate with: source venv/bin/activate"
