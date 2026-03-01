# Agentic RAG

基于检索增强生成（RAG）与 Agent 的智能问答项目，面向高校校规等文档的查询与回答，支持 SFT + GRPO 后训练。

## 功能概览

- **RAG 流程**：Query 改写与关键词提取 → 向量检索（Chroma）→ 关键词重排 → 精读筛选 → 回答生成  
- **工具与 Agent**：Query 预处理、检索、精读等封装为工具，由 LangChain Agent 调度  
- **SFT**：用轨迹数据做监督微调（VeRL，支持预分词 Parquet + LoRA）  
- **GRPO**：基于任务成功率、检索有效性与效率惩罚的奖励函数，做组相对策略优化  

## 环境与依赖

- Python 3.10+
- 主要依赖：`langchain`、`langchain-chroma`、`transformers`、`chromadb`（`pip install -r requirements.txt`）、VeRL（见 `verl/`，建议 `pip install -e ./verl`）
- 大模型：Qwen（DashScope API），需配置环境变量 `QWEN_API_KEY`

## 项目结构

```
├── main.py                 # 交互式问答入口
├── build_index.py          # 构建 Chroma 向量索引
├── build_sft_trajectories.py   # 从评估问题生成 SFT 轨迹
├── convert_trajs_to_verl_format.py
├── jsonl_to_tokenized_parquet.py
├── split_sft_train_val.py
├── data/
│   ├── internal_docs/      # 校规等原始文档
│   ├── eval/               # 评估问题 queries.txt
│   └── verl_trajs/         # 轨迹、SFT/GRPO 用 parquet
├── src/
│   ├── llm.py              # Qwen 调用
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── tools/              # query_preprocess, retrieval, reading
│   └── agents/             # RAG Agent
├── scripts/
│   ├── run_sft.sh          # SFT 训练启动
│   ├── run_grpo.sh         # GRPO 训练启动
│   ├── reward_rag.py       # GRPO 奖励函数
│   ├── build_grpo_prompt_parquet.py
│   ├── build_rm_pairwise_parquet.py
│   └── pretokenized_sft_dataset.py
├── docs/                   # 流程说明、面试问答等
└── verl/                   # VeRL 框架（SFT/GRPO）
```

## 快速开始

### 1. 构建检索索引

```bash
python build_index.py --data_dir data/internal_docs --persist_dir rag_cache/chroma_db
```

### 2. 交互问答

```bash
export QWEN_API_KEY=your_key
python main.py
```

### 3. SFT 训练（可选）

```bash
# 生成轨迹 → 转 VeRL 格式 → 分词为 parquet（见文档）
pip install -e ./verl
bash scripts/run_sft.sh
```

### 4. GRPO 训练（可选）

```bash
python scripts/build_grpo_prompt_parquet.py --queries data/eval/queries.txt
bash scripts/run_grpo.sh
```

更多细节见 `docs/GRPO_流程说明.md`、`docs/GRPO_与奖励模型说明.md`。

## 奖励设计（GRPO）

- **任务成功率**：答案正确时大额正奖励  
- **检索有效性**：回复包含关键证据时中等奖励  
- **效率惩罚**：超步数、无效工具调用给予负奖励  

实现见 `scripts/reward_rag.py`。

## License

见仓库说明。
