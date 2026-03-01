from src.llm import get_qwen_chat
from src.agents.rag_agent import RagAgent


def demo():
    llm = get_qwen_chat()
    agent = RagAgent(llm=llm)

    query = input("请输入你的问题：")
    result = agent.run(query)
    print("\n=== Agent 回答 ===")
    print(result.agent_answer)


if __name__ == "__main__":
    demo()

