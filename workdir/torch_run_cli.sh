ROOT=/mnt/workspace/LLaMA-Factory

export PYTHONPATH=$ROOT/src:$ROOT:$PYTHONPYTH
MODEL=/dev/shm/Qwen2.5-VL-72B-Instruct/Qwen/Qwen2.5-VL-72B-Instruct

FORCE_TORCHRUN=1
CONFIG=$1
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
CONFIG_NAME=$(basename "$CONFIG" "${CONFIG##*.}")

mkdir -p logs

python -m llamafactory.cli train $CONFIG 2>&1 | tee logs/run_${CONFIG_NAME}_${TIMESTAMP}.log
