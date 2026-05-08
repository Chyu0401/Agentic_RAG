# Agentic RAG

基于检索增强生成（RAG）与 Agent 的智能问答项目，面向高校校规等文档的查询与回答，支持 SFT + GRPO 后训练。

## 架构概览

项目存在两套架构：

- **部署侧（v1）**：LangChain 三工具 Agent —— query 预处理、检索、精读，由 `create_agent` 调度
- **训练侧（v2）**：Pure ReAct + 单 retrieval 工具 —— query 改写与精读内化到模型 chain-of-thought，由 VeRL multi-turn 调度（SGLang rollout + 真实工具执行）

## 功能概览

- **RAG 流程**：向量检索（Chroma）→ 关键词感知重排 → 精读筛选 / 内化推理 → 回答生成
- **SFT**：Rejection Sampling SFT（教师 K 采样 + reward 过滤），多轮 chat 格式，loss mask 自动只在 assistant token 算
- **GRPO**：multi-turn rollout 真实工具执行，结构化轨迹级 reward（任务成功率 + 检索有效性 + 效率惩罚）

## 环境与依赖

- Python 3.10+
- 主要依赖：`langchain`、`langchain-chroma`、`transformers`、`chromadb`、`sglang`，VeRL（建议 `pip install -e ./verl`）
- 教师 LLM：Qwen（DashScope API），需配置 `QWEN_API_KEY`
- 训练 GPU：6×A100-80G 单机（FSDP2）

## 项目结构

```
├── main.py                              # 交互式问答入口（v1 部署）
├── build_index.py                       # 构建 Chroma 向量索引
├── build_sft_trajectories.py            # v1：LangChain Agent 收集轨迹
├── convert_trajs_to_verl_format.py
├── jsonl_to_tokenized_parquet.py
├── split_sft_train_val.py
├── offline_eval.py                      # 上下文长度 / 无关比例评估
├── data/
│   ├── internal_docs/                   # 校规等原始文档
│   ├── eval/                            # 评估问题 queries.txt
│   └── verl_trajs/                      # 轨迹、SFT/GRPO 用 parquet
├── src/                                 # v1 部署侧（LangChain）
│   ├── llm.py                           # Qwen 调用
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── tools/                           # query_preprocess, retrieval, reading
│   └── agents/                          # RAG Agent
├── scripts/
│   ├── run_sft.sh                       # multi-turn SFT 启动
│   ├── run_grpo.sh                      # multi-turn GRPO 启动
│   ├── reward_rag.py                    # GRPO 奖励函数（结构化轨迹）
│   ├── build_grpo_prompt_parquet.py     # GRPO prompt parquet（含 Pure ReAct system prompt）
│   ├── build_sft_trajectories_react.py  # v2：Rejection Sampling 教师采样
│   ├── convert_react_trajs_to_verl_format.py  # 多轮轨迹 → SFT parquet
│   ├── v1_vs_v2_compare.py              # v1/v2 端到端对比
│   ├── eval_reasoning_efficiency.py     # 推理效率评估（GRPO 前后）
│   ├── verl_tools/                      # v2 训练工具（BaseTool 子类）
│   │   └── retrieval_tool.py
│   └── tool_config/
│       └── rag_tools.yaml               # VeRL multi-turn 工具注册
├── docs/                                # 流程说明、面试问答等
└── verl/                                # VeRL 框架（SFT/GRPO）
```

## 快速开始

### 1. 构建检索索引

```bash
python build_index.py --data_dir data/internal_docs --persist_dir rag_cache/chroma_db
```

### 2. 交互问答（v1 部署）

```bash
export QWEN_API_KEY=your_key
python main.py
```

### 3. SFT 训练（v2 multi-turn + Rejection Sampling）

```bash
# 教师采样 K=6，按 reward >= 1.0 过滤
python scripts/build_sft_trajectories_react.py --queries data/eval/queries.txt --k_per_query 6

# 多轮 chat 转 parquet（含 tools schema）
python scripts/convert_react_trajs_to_verl_format.py

# 启动 SFT
pip install -e ./verl
bash scripts/run_sft.sh
```

### 4. GRPO 训练（v2 multi-turn agentic RL）

```bash
python scripts/build_grpo_prompt_parquet.py --queries data/eval/queries.txt
bash scripts/run_grpo.sh
```

### 5. v1 vs v2 端到端对比

```bash
python scripts/v1_vs_v2_compare.py --trajs data/verl_trajs/trajs.jsonl --limit 50
# 输出在 data/eval/v1_vs_v2/summary.md
```

## 奖励设计（GRPO）

`scripts/reward_rag.py` 基于 multi-turn 结构化轨迹解析（chat template marker），三段奖励叠加：

- **任务成功率**：最后一条 assistant 答案命中 ground_truth → +1.0
- **检索有效性**：tool 块返回内容包含 key_evidence → +0.4（**只查 tool 块**，防 reward hacking）
- **效率惩罚**：tool 调用 > 3 次每次 -0.1

## License

见仓库说明。
