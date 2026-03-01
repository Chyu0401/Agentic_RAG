"""
RAG / Agentic RAG 的 GRPO 奖励函数。
包含：任务成功率奖励（大额正奖励）、检索有效性奖励（中等奖励）、效率惩罚（无效工具调用 / 超步）。
供 VeRL 通过 custom_reward_function.path/name 加载。
"""
from __future__ import annotations

import re
from typing import Any, List, Optional

# 奖励/惩罚系数（可经 extra_info 或 kwargs 覆盖）
REWARD_TASK_SUCCESS = 1.0
REWARD_RETRIEVAL_EFFECTIVE = 0.4
PENALTY_PER_EXTRA_STEP = 0.1
PENALTY_PER_INVALID_CALL = 0.15
MAX_STEPS_NO_PENALTY = 8


def _normalize(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_final_answer(solution_str: str, max_tail_chars: int = 800) -> str:
    if not solution_str or len(solution_str) < max_tail_chars:
        return (solution_str or "").strip()
    return solution_str[-max_tail_chars:].strip()


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
    """从 ground_truth 或 extra_info 取出「关键证据」片段（用于检索有效性）。"""
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
        # 用 keywords 作为证据代理
        kw = ground_truth.get("keywords") or []
        if kw:
            return [str(k).strip() for k in kw if k]
    return []


def _count_steps_and_invalid_calls(solution_str: str) -> tuple[int, int]:
    """从回复文本中估计步数及无效工具调用次数。
    步数：匹配 Action/Observation/Thought、或「调用」「步骤」等。
    无效调用：重复的 Action 行、或 Observation 中常见失败/空结果。
    """
    if not solution_str or not solution_str.strip():
        return 0, 0
    text = solution_str
    # 步数：成对 Action + Observation 或单次 Thought 算一步
    action_pat = re.compile(r"(?:Action|行动|调用)\s*[：:]\s*\w+", re.I)
    observation_pat = re.compile(r"(?:Observation|观察|结果)\s*[：:]", re.I)
    thought_pat = re.compile(r"(?:Thought|思考)\s*[：:]", re.I)
    steps = max(
        len(action_pat.findall(text)),
        len(observation_pat.findall(text)),
        len(thought_pat.findall(text)),
    )
    if steps == 0:
        # 无明确标记时，用「步骤」「Step」等粗略估计
        step_mentions = re.findall(r"(?:步骤|Step)\s*[：:]?\s*\d+", text, re.I)
        steps = min(len(step_mentions), 20)
    # 无效调用：重复的 Action 行（同一工具名出现多次且中间无 Observation）
    action_lines = action_pat.findall(text)
    invalid = 0
    seen = []
    for a in action_lines:
        key = _normalize(a)[:50]
        if key in seen:
            invalid += 1
        else:
            seen.append(key)
    # 观测到「未找到」「无结果」「error」等计为可能无效
    fail_obs = re.compile(
        r"(?:Observation|观察|结果)\s*[：:][^\n]*(?:未找到|无结果|error|失败|无法)",
        re.I,
    )
    invalid += len(fail_obs.findall(text))
    return min(steps, 50), max(0, invalid)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    method: str = "flexible",
    reward_task_success: float = REWARD_TASK_SUCCESS,
    reward_retrieval_effective: float = REWARD_RETRIEVAL_EFFECTIVE,
    penalty_per_extra_step: float = PENALTY_PER_EXTRA_STEP,
    penalty_per_invalid_call: float = PENALTY_PER_INVALID_CALL,
    max_steps_no_penalty: int = MAX_STEPS_NO_PENALTY,
    **kwargs: Any,
) -> float:
    """基于任务成功率、检索有效性与效率惩罚的 RAG 奖励。

    - 任务成功率：最终答案正确则给予大额正奖励（reward_task_success，默认 1.0）。
    - 检索有效性：回复中包含标准答案的关键证据片段则给予中等奖励（reward_retrieval_effective，默认 0.4）。
    - 效率惩罚：步数超过 max_steps_no_penalty 每多一步扣 penalty_per_extra_step；每次无效工具调用扣 penalty_per_invalid_call。
    """
    extra_info = extra_info or {}
    if data_source not in ("agentic_rag", "rag"):
        raise NotImplementedError(f"Reward not implemented for data_source={data_source!r}")

    # 解析 ground_truth
    if isinstance(ground_truth, dict):
        ref_str = _normalize((ground_truth.get("answer") or ground_truth.get("ref") or ""))
        keywords = ground_truth.get("keywords") or []
    else:
        ref_str = _normalize(str(ground_truth))
        keywords = []

    answer_span = _extract_final_answer(solution_str)
    norm_answer = _normalize(answer_span)
    full_norm = _normalize(solution_str or "")

    reward = 0.0

    # 1）任务成功率奖励：答案正确则大额正奖励
    if ref_str or keywords:
        if _is_task_correct(norm_answer, ref_str, keywords, method):
            reward += reward_task_success

    # 2）检索有效性奖励：回复中出现关键证据则中等奖励（与任务成功独立，可叠加）
    key_evidence = _get_key_evidence(ground_truth, extra_info)
    if key_evidence:
        full_lower = full_norm.lower()
        hit = sum(1 for e in key_evidence if e and e.lower() in full_lower)
        if hit >= max(1, len(key_evidence) * 0.5):
            reward += reward_retrieval_effective

    # 3）效率惩罚：超步数 + 无效工具调用
    steps, invalid_calls = _count_steps_and_invalid_calls(solution_str)
    if steps > max_steps_no_penalty:
        reward -= (steps - max_steps_no_penalty) * penalty_per_extra_step
    reward -= invalid_calls * penalty_per_invalid_call

    return round(reward, 4)
