import os
from typing import Optional

from langchain_community.chat_models import ChatTongyi


def get_qwen_chat(
    model_name: str = "qwen-plus",
    api_key: Optional[str] = None,
) -> ChatTongyi:
    if api_key is None:
        api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 Qwen API Key，请在环境变量 QWEN_API_KEY 中配置，"
            "或调用 get_qwen_chat(api_key=...) 显式传入。"
        )

    model = ChatTongyi(
        model=model_name,
        dashscope_api_key=api_key,
    )
    return model

