#!/bin/bash
set -e

# 从项目根目录 (Agentic_RAG) 运行，保证 data.custom_cls.path 和 data 路径正确
cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"

# 未用 "pip install -e ./verl" 时设置 PYTHONPATH；若报 importlib.metadata StopIteration，请改用 pip install -e ./verl
export PYTHONPATH="${PROJECT_ROOT}/verl:${PYTHONPATH:-}"

# 4 卡训练，按实际修改
nproc_per_node=2

# 预分词 parquet 路径（与 jsonl_to_tokenized_parquet 产出一致）
TRAIN_PARQUET="${TRAIN_PARQUET:-$PROJECT_ROOT/data/verl_trajs/trajs_verl_sft_train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-$PROJECT_ROOT/data/verl_trajs/trajs_verl_sft_val.parquet}"

# 基座模型：需与做预分词时用的 tokenizer 一致
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"

torchrun \
  --nnodes=1 \
  --nproc_per_node=$nproc_per_node \
  -m verl.trainer.sft_trainer \
  data.train_files="$TRAIN_PARQUET" \
  data.val_files="$VAL_PARQUET" \
  data.micro_batch_size_per_gpu=4 \
  data.max_length=4096 \
  model.path="$MODEL_PATH" \
  model.override_config.attn_implementation=eager \
  model.lora_rank=8 \
  model.lora_alpha=16 \
  model.target_modules=all-linear \
  trainer.project_name=rag-sft \
  trainer.experiment_name=trajs-sft-qwen-lora \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=$nproc_per_node \
  trainer.total_epochs=3 \
  trainer.logger='["console"]'
