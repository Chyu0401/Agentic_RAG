#!/usr/bin/env python3
"""
将 data/verl_trajs/trajs.jsonl 转为 VeRL SFT 可直接使用的格式。

输出每条样本包含 VeRL 所需的 messages 键：
- messages: [ {"role": "user", "content": query}, {"role": "assistant", "content": 工具调用序列+最终回答} ]

同时保留 query / trajectory / final_answer / task_success 等字段，便于做 RL 或分析。

用法：
  python convert_trajs_to_verl_format.py
  python convert_trajs_to_verl_format.py --input ./data/verl_trajs/trajs.jsonl --output ./data/verl_trajs/trajs_verl_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _tool_output_to_str(tool_output: Any) -> str:
    """将 tool_output（可能为 dict）转为字符串。"""
    if isinstance(tool_output, str):
        return tool_output
    if isinstance(tool_output, dict):
        return json.dumps(tool_output, ensure_ascii=False)
    return str(tool_output)


def _build_assistant_content(history: list, final_answer: str) -> str:
    """将轨迹 + 最终回答拼成 assistant 的 content，供 SFT 学习工具调用与回答。"""
    parts = []
    for step in history:
        action = step.get("action", {})
        tool = action.get("tool", "")
        tool_input = action.get("tool_input", {})
        tool_output = action.get("tool_output")
        next_str = _tool_output_to_str(tool_output) if tool_output is not None else ""
        parts.append(f"Action: {tool}({json.dumps(tool_input, ensure_ascii=False)})\nObservation: {next_str}")
    parts.append(f"Answer: {final_answer}")
    return "\n\n".join(parts)


def convert_one(raw: dict) -> dict:
    """
    将一条原始轨迹转为 VeRL SFT 所需格式：含 messages（user + assistant），
    并保留 query / trajectory / final_answer / task_success 等字段。
    """
    query = raw.get("query", "")
    history = raw.get("history", [])
    final_answer = raw.get("final_answer", "")
    task_success = raw.get("task_success")

    trajectory = []
    for i, step in enumerate(history):
        obs = step.get("observation", "")
        action = step.get("action", {})
        tool_output = action.get("tool_output")
        next_obs = _tool_output_to_str(tool_output) if tool_output is not None else ""
        reward = None
        if task_success is not None and i == len(history) - 1:
            reward = 1.0 if task_success else 0.0
        trajectory.append({
            "observation": obs,
            "action": action,
            "reward": reward,
            "next_observation": next_obs,
        })

    assistant_content = _build_assistant_content(history, final_answer)
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": assistant_content},
    ]

    return {
        "messages": messages,
        "query": query,
        "trajectory": trajectory,
        "final_answer": final_answer,
        "task_success": task_success,
        "eval": task_success,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="将 Agent 轨迹 JSONL 转为 VeRL SFT 用格式（含 messages）")
    parser.add_argument(
        "--input",
        type=str,
        default="./data/verl_trajs/trajs.jsonl",
        help="原始 trajs.jsonl 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/verl_trajs/trajs_verl_sft.jsonl",
        help="输出 VeRL 格式 JSONL 路径",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        raise FileNotFoundError(f"输入文件不存在: {inp}")

    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with inp.open("r", encoding="utf-8") as fr, out.open("w", encoding="utf-8") as fw:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            converted = convert_one(raw)
            fw.write(json.dumps(converted, ensure_ascii=False) + "\n")
            n += 1

    print(f"已转换 {n} 条轨迹 -> {out}")


if __name__ == "__main__":
    main()
