"""VeRL multi-turn 训练专用工具集（Pure ReAct 模式，仅暴露 retrieval）。

与 src/tools/ 目录的区别：
- src/tools/：LangChain Agent 部署用，依赖 ChatPromptTemplate 与 Qwen API
- scripts/verl_tools/：VeRL multi-turn rollout 用，BaseTool 子类，纯函数接口
"""
