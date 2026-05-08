#!/bin/bash
# Multi-turn SFT（rejection-sampling 后的 ReAct 轨迹），6 卡 80G A100/H100 单机
# 走 verl.trainer.fsdp_sft_trainer + MultiTurnSFTDataset
# loss mask 自动只在 assistant token 上算（tool observation / system / user 不算 loss）
#
# 运行前依次执行：
#   1) export QWEN_API_KEY=xxx
#   2) python scripts/build_sft_trajectories_react.py --queries_file data/eval/queries.txt
#   3) python scripts/convert_react_trajs_to_verl_format.py
#   4) bash scripts/run_sft.sh
set -e

cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/verl:${PYTHONPATH:-}"

# 6 卡单机
nproc_per_node=6

# Multi-turn SFT 用的 parquet（convert_react_trajs_to_verl_format.py 的输出）
TRAIN_PARQUET="${TRAIN_PARQUET:-$PROJECT_ROOT/data/verl_trajs/sft_react_train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-$PROJECT_ROOT/data/verl_trajs/sft_react_val.parquet}"

# 基座模型：与 GRPO 阶段一致
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"

SAVE_PATH="${SAVE_PATH:-$PROJECT_ROOT/checkpoints/sft_react}"

# === 关键 multi-turn 配置 ===
# data.multiturn.enable=true：启用 MultiTurnSFTDataset（loss mask 自动只算 assistant token）
# data.multiturn.messages_key=messages：parquet 里 messages 列的字段名
# data.multiturn.tools_key=tools：parquet 里 tools 列（function schema list）字段名
#                                  → chat template 会自动把 tools 注入 system prompt 里
#
# === 长度 / batch ===
# max_length=4096：覆盖完整多轮轨迹（system + user + multi-round assistant + tool obs + final answer）
# micro_batch_size_per_gpu=2：4B 模型 + bf16 + 4096 seq + LoRA，单卡能装下
# global batch = 2 * 6 = 12，gradient accumulation 由 trainer 控制
#
# === LoRA ===
# 沿用原 SFT 设置：rank=8, alpha=16, target_modules=all-linear
# LoRA 训练显存友好，4B 全量微调 6 卡也行但没必要
torchrun \
  --nnodes=1 \
  --nproc_per_node=$nproc_per_node \
  -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.max_length=4096 \
    data.micro_batch_size_per_gpu=2 \
    data.truncation=error \
    model.partial_pretrain="$MODEL_PATH" \
    model.lora_rank=8 \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    model.enable_gradient_checkpointing=True \
    optim.lr=1e-5 \
    optim.weight_decay=0.01 \
    optim.warmup_steps_ratio=0.03 \
    use_remove_padding=true \
    trainer.default_local_dir="$SAVE_PATH" \
    trainer.project_name=rag-sft \
    trainer.experiment_name=react-multiturn-sft-qwen3-4b \
    trainer.logger='["console"]' \
    trainer.total_epochs=3 \
    trainer.save_freq=200 \
    trainer.test_freq=100 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$nproc_per_node "$@"

# === OOM 调参顺序 ===
# 1. data.micro_batch_size_per_gpu 2 → 1
# 2. data.max_length 4096 → 3072 → 2048（注意可能导致 truncation=error 报错，要么改成 truncation=right）
# 3. 加 ulysses_sequence_parallel_size=2（开 sequence parallel，省 activation 显存）
# 4. model.lora_rank 8 → 4（极端节省）
#
# === 烟雾测试 ===
# 加 trainer.total_training_steps=2 跑两步，看：
# - loss 是否合理（初始应该几 → 几十，Qwen3-4B base 在 ReAct 数据上 cross-entropy）
# - 显存是否爆
# - tools 列是否被正确加载（看日志 "tools list" 输出）
# bash scripts/run_sft.sh trainer.total_training_steps=2

# === 训完之后 ===
# checkpoint 输出在 $SAVE_PATH/global_step_*/
# 用于 GRPO 时把 actor_rollout_ref.model.path 指向训完的目录即可
# （注意 LoRA 训练只保存 adapter，需要先 merge 回 base 模型再喂 GRPO，
#  或直接 GRPO 时也指定 LoRA：actor_rollout_ref.model.lora_rank=8 这种）