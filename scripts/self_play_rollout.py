"""对已验证的 Self-Play 问题采样 G 个解答，生成 GRPO rollout。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grader.drgrpo_grader import r1_zero_reward_fn  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Play GRPO rollout")
    parser.add_argument("--model", default="models/grpo_fast")
    parser.add_argument("--input", default="data/self_play_taskbook_smoke_v2.jsonl")
    parser.add_argument("--output", default="data/self_play_rollout_smoke.jsonl")
    parser.add_argument("--prompt-file", default="prompts/r1_zero.prompt")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-tokens", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--grpo-step", type=int, default=0)
    parser.add_argument(
        "--drop-uniform-groups",
        action="store_true",
        help="只写入同时含 reward=0 和 reward=1 的组；默认保留全部组",
    )
    return parser.parse_args()


def load_items(path: Path, max_questions: int | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = row.get("question") or row.get("problem")
            ground_truth = row.get("ground_truth")
            if ground_truth is None:
                ground_truth = row.get("answer")
            if not question or ground_truth is None:
                continue
            normalized = " ".join(str(question).lower().split())
            if normalized in seen:
                continue
            seen.add(normalized)
            items.append(
                {
                    "input_idx": line_idx,
                    "source_idx": row.get("source_idx", line_idx),
                    "self_play_id": row.get("self_play_id", f"self_play_{line_idx:06d}"),
                    "question": str(question).strip(),
                    "ground_truth": ground_truth,
                }
            )
            if max_questions is not None and len(items) >= max_questions:
                break
    if not items:
        raise RuntimeError("输入文件中没有 question/problem + ground_truth/answer")
    return items


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def batched(values: list[Any], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield start, values[start : start + batch_size]


def main() -> None:
    args = parse_args()
    if args.group_size < 2:
        raise ValueError("--group-size 必须至少为 2")

    from vllm import LLM, SamplingParams

    items = load_items(Path(args.input), args.max_questions)
    prompt_template = Path(args.prompt_file).read_text(encoding="utf-8")
    prompts = [
        prompt_template.replace("{question}", item["question"]) for item in items
    ]

    print(f"[model] loading: {args.model}")
    print(f"[rollout] questions={len(items)}, G={args.group_size}")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        n=args.group_size,
        temperature=args.temperature,
        top_p=args.top_p,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    all_rows: list[dict[str, Any]] = []
    kept_rows: list[dict[str, Any]] = []
    groups_all_correct = 0
    groups_all_wrong = 0
    groups_mixed = 0
    formatted = 0
    correct = 0

    for start, prompt_batch in batched(prompts, args.batch_size):
        request_outputs = llm.generate(prompt_batch, sampling_params)
        for local_idx, request_output in enumerate(request_outputs):
            item_idx = start + local_idx
            item = items[item_idx]
            prompt_token_ids = list(request_output.prompt_token_ids)
            group_id = (
                f"self_play_step_{args.grpo_step:04d}_group_{item_idx:04d}"
            )
            group_rows: list[dict[str, Any]] = []
            rewards: list[float] = []

            for sample_idx, completion in enumerate(request_output.outputs):
                response = completion.text
                reward_info = r1_zero_reward_fn(
                    response,
                    item["ground_truth"],
                    fast=True,
                )
                reward = float(reward_info["reward"])
                rewards.append(reward)
                formatted += int(reward_info["format_reward"] == 1.0)
                correct += int(reward_info["answer_reward"] == 1.0)
                response_token_ids = list(completion.token_ids)
                group_rows.append(
                    {
                        "grpo_step": args.grpo_step,
                        "group_id": group_id,
                        "source_idx": item["source_idx"],
                        "self_play_id": item["self_play_id"],
                        "sample_idx": sample_idx,
                        "question": item["question"],
                        "final_answer": item["ground_truth"],
                        "prompt": prompt_batch[local_idx],
                        "response": response,
                        "prompt_token_ids": prompt_token_ids,
                        "response_token_ids": response_token_ids,
                        "prompt_length": len(prompt_token_ids),
                        "response_length": len(response_token_ids),
                        "format_reward": float(reward_info["format_reward"]),
                        "answer_reward": float(reward_info["answer_reward"]),
                        "reward": reward,
                        "policy_model": args.model,
                        "finish_reason": getattr(completion, "finish_reason", None),
                    }
                )

            mean_reward = sum(rewards) / len(rewards)
            for row in group_rows:
                row["advantage"] = row["reward"] - mean_reward

            all_rows.extend(group_rows)
            if all(reward == 1.0 for reward in rewards):
                groups_all_correct += 1
                group_type = "all_correct"
            elif all(reward == 0.0 for reward in rewards):
                groups_all_wrong += 1
                group_type = "all_wrong"
            else:
                groups_mixed += 1
                group_type = "mixed"

            for row in group_rows:
                row["group_mean_reward"] = mean_reward
                row["group_type"] = group_type
            if not args.drop_uniform_groups or group_type == "mixed":
                kept_rows.extend(group_rows)

        done = min(start + len(prompt_batch), len(items))
        print(f"[rollout] processed {done}/{len(items)} questions")

    output_path = Path(args.output)
    write_jsonl(output_path, kept_rows)
    stats = {
        "model": args.model,
        "input": args.input,
        "questions": len(items),
        "group_size": args.group_size,
        "rollouts_generated": len(all_rows),
        "rollouts_saved": len(kept_rows),
        "answer_accuracy": correct / len(all_rows) if all_rows else 0.0,
        "format_rate": formatted / len(all_rows) if all_rows else 0.0,
        "groups_all_correct": groups_all_correct,
        "groups_all_wrong": groups_all_wrong,
        "groups_mixed": groups_mixed,
        "drop_uniform_groups": args.drop_uniform_groups,
    }
    print("[stats]")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[save] rollout: {output_path}")
    sys.stdout.flush()

    del llm
    gc.collect()
    os._exit(0)


if __name__ == "__main__":
    main()
