#!/bin/bash
# GRPO 后训练（multi-turn agentic RL，6 卡 80G A100/H100 单机）
# 工具：Pure ReAct，仅 retrieval_augment 一个工具（详见 scripts/tool_config/rag_tools.yaml）
# 推理引擎：SGLang（multi-turn 主路径，替代 vLLM）
# 运行前：
#   1) pip install -e ./verl  &&  pip install sglang（已确认）
#   2) python scripts/build_grpo_prompt_parquet.py --queries data/eval/queries.txt（或带 --trajs）
#   3) 确保 ./rag_cache/chroma_db 已 build_index.py 构建好
set -e

cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"
# 把项目根目录加到 PYTHONPATH，让 verl 能 import scripts.verl_tools.* 与 src.*
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/verl:${PYTHONPATH:-}"

# 6 卡单机配置
nproc_per_node=6

TRAIN_PARQUET="${TRAIN_PARQUET:-$PROJECT_ROOT/data/verl_trajs/grpo_prompts_train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-$PROJECT_ROOT/data/verl_trajs/grpo_prompts_val.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
REWARD_PATH="${PROJECT_ROOT}/scripts/reward_rag.py"
TOOL_CONFIG="${PROJECT_ROOT}/scripts/tool_config/rag_tools.yaml"

# === Batch 整除约束（6 卡） ===
#   train_batch_size=96, rollout.n=4 → 96*4=384 trajectory/step
#   ppo_mini_batch_size=48 → 384/48=8 PPO 更新/step
#   ppo_micro_batch_size_per_gpu=4, n_gpus=6 → 48/(4*6)=2 grad accum/PPO 更新
# 含因子 3 的数（96, 48）才能被 6 整除；原 128/64 不行
#
# === 长度配置 ===
# max_response_length=4096：multi-turn ReAct 多轮（最多 3 次 retrieval × ~1500 token 文档段）
#                          + 模型 thinking + 最终答案，3072 偏紧，4096 安全
# max_prompt_length=2048：system prompt（Pure ReAct 指令较长）+ user query + tool 注入空间
#
# === Multi-turn 关键开关 ===
# data.return_raw_chat=True：保留 raw chat 结构供 SGLang 多轮 rollout 使用
# rollout.name=sglang + mode=async：multi-turn 主路径，vllm 不行
# rollout.multi_turn.enable=True + tool_config_path：启用工具调用循环
# rollout.multi_turn.max_assistant_turns=4：最多 4 次 assistant 生成（含最终回答）

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="[\"$TRAIN_PARQUET\"]" \
  data.val_files="[\"$VAL_PARQUET\"]" \
  data.train_batch_size=96 \
  data.max_prompt_length=2048 \
  data.max_response_length=4096 \
  data.prompt_key=prompt \
  data.return_raw_chat=True \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  reward.custom_reward_function.path="$REWARD_PATH" \
  reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=48 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_p=0.9 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name=rag-grpo \
  trainer.experiment_name=rag-grpo-multiturn-6gpu \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=$nproc_per_node \
  trainer.save_freq=100 \
  trainer.test_freq=50 \
  trainer.total_epochs=5 "$@"

# === OOM 调参顺序 ===
# 1. rollout.gpu_memory_utilization 0.45 → 0.4 → 0.35
# 2. data.max_response_length 4096 → 3072
# 3. rollout.multi_turn.max_assistant_turns 4 → 3
# 4. rollout.n 4 → 3（注意 train_batch 同步调到 96，96*3=288，mini 调到 36 或 48 看整除）
# 5. ppo_micro_batch_size_per_gpu 4 → 2（mini 同步调到 24：24/(2*6)=2）
# 6. retrieval_tool 的 max_chars_per_chunk 800 → 500（在 rag_tools.yaml 里改）
#
# === 已知坑 ===
# 1. RetrievalTool 在 Ray worker 里第一次调用会加载 Chroma + embedding 模型，第一条 trajectory 会慢
# 2. tool_config_path 必须用绝对路径，相对路径在 Ray worker 里可能找不到
# 3. PYTHONPATH 必须包含项目根目录（让 verl 能 import scripts.verl_tools.*）
# 4. 第一次跑前先单步测：用 build_grpo_prompt_parquet.py 出一个只有 1-2 条的 train.parquet，
#    再把 trainer.total_training_steps=2 加到命令里，验证轨迹格式 + reward 曲线 + tool 真的被调用