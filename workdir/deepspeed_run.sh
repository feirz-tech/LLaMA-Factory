ROOT=/mnt/workspace/demos/llama_factory_qwen2vl/LLaMA-Factory
export PYTHONPATH=$ROOT:$PYTHONPYTH
MODEL=/mnt/workspace/demos/llama_factory_qwen2vl/LLaMA-Factory/Qwen2-VL-2B-Instruct

deepspeed --num_gpus 2 $ROOT/src/train.py \
  --deepspeed $ROOT/examples/deepspeed/ds_z0_config.json \
  --stage sft \
  --model_name_or_path $MODEL  \
  --do_train \
  --preprocessing_num_workers 8 \
  --dataset_dir $ROOT/data \
  --dataset train \
  --template qwen2_vl \
  --cutoff_len 4096 \
  --finetuning_type full \
  --output_dir saves/qwenvl \
  --overwrite_cache \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --max_grad_norm 1.0 \
  --save_steps 500 \
  --save_only_model True \
  --learning_rate 1e-6 \
  --num_train_epochs 10.0 \
  --warmup_steps 0 \
  --optim adamw_torch \
  --packing False \
  --plot_loss True \
  --bf16
