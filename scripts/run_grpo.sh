#!/bin/bash
# GRPO 后训练：使用 function-based reward（RAG 自定义 compute_score），无需单独 RM 模型。
# 运行前：1）pip install -e ./verl  2）生成 GRPO prompt parquet：python scripts/build_grpo_prompt_parquet.py
set -e

cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"
export PYTHONPATH="${PROJECT_ROOT}/verl:${PYTHONPATH:-}"

nproc_per_node=4
TRAIN_PARQUET="${TRAIN_PARQUET:-$PROJECT_ROOT/data/verl_trajs/grpo_prompts_train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-$PROJECT_ROOT/data/verl_trajs/grpo_prompts_val.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
REWARD_PATH="${PROJECT_ROOT}/scripts/reward_rag.py"

# 多卡请用：torchrun --nnodes=1 --nproc_per_node=$nproc_per_node -m verl.trainer.main_ppo ...
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="[\"$TRAIN_PARQUET\"]" \
  data.val_files="[\"$VAL_PARQUET\"]" \
  data.train_batch_size=128 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.prompt_key=prompt \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  reward.custom_reward_function.path="$REWARD_PATH" \
  reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_p=0.9 \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name=rag-grpo \
  trainer.experiment_name=rag-grpo-qwen-lora \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=$nproc_per_node \
  trainer.save_freq=100 \
  trainer.test_freq=50 \
  trainer.total_epochs=5 "$@"
