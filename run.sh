#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"

python -m pip install --disable-pip-version-check -q -r requirements-repro.txt
torchrun --standalone --nproc_per_node=4 run_experiment.py --config experiment.json
