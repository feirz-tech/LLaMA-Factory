#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH=$ROOT/src:$ROOT:$PYTHONPYTH

CONFIG=$1
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
CONFIG_NAME=$(basename "$CONFIG" "${CONFIG##*.}")

mkdir -p logs

accelerate launch \
--config_file accelerator_config.yaml \
$ROOT/src/train.py $CONFIG 2>&1 | tee logs/run_${CONFIG_NAME}_${TIMESTAMP}.log
