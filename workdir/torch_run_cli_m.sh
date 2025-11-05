#!/bin/bash

export NNODES=$2
export NODE_RANK=$RANK
export NPROC_PER_NODE=8
# export NCCL_DEBUG=WARN


# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# LLaMA-Factory 根目录是脚本目录的上一级
ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH=$ROOT/src:$ROOT:$PYTHONPYTH

export FORCE_TORCHRUN=1
CONFIG=$1
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
CONFIG_NAME=$(basename "$CONFIG" "${CONFIG##*.}")

mkdir -p logs

python -m llamafactory.cli train $CONFIG 2>&1 | tee logs/run_${CONFIG_NAME}_${TIMESTAMP}.log
