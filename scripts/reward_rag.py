"""Agentic RAG 的 GRPO 奖励函数（multi-turn 版）。

与旧版的关键区别：旧版从一段 ReAct 文本里**正则解析** Action/Observation，
新版基于 VeRL multi-turn 真实生成的**结构化轨迹** —— solution_str 是模型自己生成
的全部 token（含 chat template marker 和 <tool_call> 标签），tool 返回内容
则被 chat template 包裹在 <|im_start|>tool ... <|im_end|> 段里。

奖励三段：
1. 任务成功率：从最后一条 assistant message 取最终答案，与 ground truth 比对
2. 检索有效性：检测 tool 返回内容里是否包含 ground truth 的 key_evidence
3. 效率惩罚：tool_call 次数超阈值则按多调一次减一份惩罚

供 VeRL 通过 reward.custom_reward_function.path / name 加载。
"""
from __future__ import annotations

import re
from typing import Any, List, Optional

REWARD_TASK_SUCCESS = 1.0
REWARD_RETRIEVAL_EFFECTIVE = 0.4
PENALTY_PER_EXTRA_STEP = 0.1
MAX_STEPS_NO_PENALTY = 3   # Pure ReAct 通常 1-3 次检索足够，> 3 开始惩罚

# 匹配 chat template 的 marker
# 适配 Qwen3 / Qwen2.5 系列：<|im_start|>role ... <|im_end|>
_ASSISTANT_BLOCK = re.compile(
    r"<\|im_start\|>assistant\s*\n(.*?)<\|im_end\|>", re.DOTALL
)
_TOOL_BLOCK = re.compile(
    r"<\|im_start\|>tool\s*\n(.*?)<\|im_end\|>", re.DOTALL
)
# tool_call 标签（Qwen 风格）
_TOOL_CALL_TAG = re.compile(r"<tool_call>", re.IGNORECASE)


def _normalize(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip())


def _extract_final_answer(solution_str: str) -> str:
    """取最后一条 assistant message。这就是模型给出的最终回答。

    multi-turn 下 solution_str 形如：
        <|im_start|>assistant\nThought:...\n<tool_call>...</tool_call><|im_end|>
        <|im_start|>tool\n<retrieved>\n<|im_end|>
        <|im_start|>assistant\nThought:...\nFinal answer:...<|im_end|>
    """
    matches = _ASSISTANT_BLOCK.findall(solution_str or "")
    if matches:
        return matches[-1].strip()
    # fallback：单轮或格式异常时取末尾
    return (solution_str or "").strip()[-1500:]


def _extract_tool_results(solution_str: str) -> List[str]:
    """取所有 tool 返回内容。"""
    return _TOOL_BLOCK.findall(solution_str or "")


def _count_tool_calls(solution_str: str) -> int:
    """统计 tool_call 数。优先用 <tool_call> 标签数；fallback 用 tool 块数（粗估）。"""
    tag_count = len(_TOOL_CALL_TAG.findall(solution_str or ""))
    if tag_count > 0:
        return tag_count
    # fallback：tool 块数应该等于 tool_call 数
    return len(_TOOL_BLOCK.findall(solution_str or ""))


def _is_task_correct(
    norm_answer: str,
    ref_str: str,
    keywords: List[str],
    method: str = "flexible",
) -> bool:
    if method == "strict":
        return bool(ref_str and norm_answer == ref_str)
    if ref_str and ref_str in norm_answer:
        return True
    if keywords:
        norm_lower = norm_answer.lower()
        hit = sum(1 for k in keywords if k and str(k).strip().lower() in norm_lower)
        if hit >= max(1, len(keywords) * 0.5):
            return True
    return False


def _get_key_evidence(ground_truth: Any, extra_info: dict) -> List[str]:
    """从 ground_truth / extra_info 取出关键证据片段（用于检索有效性评估）。"""
    evidence = extra_info.get("key_evidence")
    if isinstance(evidence, list):
        return [str(x).strip() for x in evidence if x]
    if isinstance(evidence, str) and evidence.strip():
        return [evidence.strip()]
    if isinstance(ground_truth, dict):
        ev = ground_truth.get("key_evidence") or ground_truth.get("evidence")
        if isinstance(ev, list):
            return [str(x).strip() for x in ev if x]
        if isinstance(ev, str) and ev.strip():
            return [ev.strip()]
        kw = ground_truth.get("keywords") or []
        if kw:
            return [str(k).strip() for k in kw if k]
    return []


def _check_retrieval_evidence(tool_results: List[str], key_evidence: List[str]) -> bool:
    """检测 tool 返回里是否包含足够多的关键证据。"""
    if not tool_results or not key_evidence:
        return False
    full_tool_text = "\n".join(tool_results).lower()
    hit = sum(1 for e in key_evidence if e and e.lower() in full_tool_text)
    return hit >= max(1, len(key_evidence) * 0.5)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    method: str = "flexible",
    reward_task_success: float = REWARD_TASK_SUCCESS,
    reward_retrieval_effective: float = REWARD_RETRIEVAL_EFFECTIVE,
    penalty_per_extra_step: float = PENALTY_PER_EXTRA_STEP,
    max_steps_no_penalty: int = MAX_STEPS_NO_PENALTY,
    **kwargs: Any,
) -> float:
    """multi-turn Agentic RAG 的轨迹级奖励。

    - 任务成功率：最终 assistant 回复正确则 +reward_task_success（默认 1.0）
    - 检索有效性：tool 返回包含 key_evidence 则 +reward_retrieval_effective（默认 0.4）
    - 效率惩罚：tool_call 次数 > max_steps_no_penalty，每多一次 -penalty_per_extra_step
    """
    extra_info = extra_info or {}
    if data_source not in ("agentic_rag", "rag"):
        raise NotImplementedError(f"Reward not implemented for data_source={data_source!r}")

    # 解析 ground_truth
    if isinstance(ground_truth, dict):
        ref_str = _normalize(ground_truth.get("answer") or ground_truth.get("ref") or "")
        keywords = ground_truth.get("keywords") or []
    else:
        ref_str = _normalize(str(ground_truth))
        keywords = []

    final_answer_raw = _extract_final_answer(solution_str)
    norm_answer = _normalize(final_answer_raw)
    tool_results = _extract_tool_results(solution_str)
    n_tool_calls = _count_tool_calls(solution_str)

    reward = 0.0

    # 1）任务成功率
    if ref_str or keywords:
        if _is_task_correct(norm_answer, ref_str, list(keywords), method):
            reward += reward_task_success

    # 2）检索有效性
    key_evidence = _get_key_evidence(ground_truth, extra_info)
    if _check_retrieval_evidence(tool_results, key_evidence):
        reward += reward_retrieval_effective

    # 3）效率惩罚（仅当有调用且超过阈值）
    if n_tool_calls > max_steps_no_penalty:
        reward -= (n_tool_calls - max_steps_no_penalty) * penalty_per_extra_step

    return round(reward, 4)
