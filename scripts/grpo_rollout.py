from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grader.drgrpo_grader import r1_zero_reward_fn
from utils.grpo import compute_group_advantages


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate one GRPO rollout batch"
    )

    p.add_argument(
        "--model",
        type=str,
        default=str(ROOT / "models" / "dpo_beta01"),
        help="当前 policy，也是本轮 rollout 的 old policy",
    )
    p.add_argument(
        "--questions-path",
        type=str,
        default=str(ROOT / "data" / "math_sft_train.jsonl"),
    )
    p.add_argument(
        "--num-questions",
        type=int,
        default=32,
        help="本轮 rollout 抽取的问题数",
    )
    p.add_argument(
        "--group-size",
        "--G",
        dest="group_size",
        type=int,
        default=8,
        help="每道题生成的回答数量",
    )

    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--min-tokens", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=1024)

    p.add_argument(
        "--grpo-step",
        type=int,
        default=0,
        help="当前 GRPO outer step，用于生成不同随机种子和 group_id",
    )
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="一次送给 vLLM 的问题数",
    )

    p.add_argument(
        "--normalize-advantages",
        action="store_true",
        help="使用 (reward-mean)/(std+eps)；默认仅减去组内均值",
    )
    p.add_argument(
        "--advantage-eps",
        type=float,
        default=1e-6,
    )

    p.add_argument("--fast", action="store_true", default=True)
    p.add_argument(
        "--no-fast",
        dest="fast",
        action="store_false",
    )

    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
    )

    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="默认写入 data/grpo_rollout_step_XXXX.jsonl",
    )

    return p.parse_args()


def load_questions(
    path: Path,
    num_questions: int,
    seed: int,
    grpo_step: int,
) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"找不到问题数据: {path}")

    items = []

    with path.open("r", encoding="utf-8") as f:
        for source_idx, line in enumerate(f):
            if not line.strip():
                continue

            row = json.loads(line)

            required_fields = {
                "question",
                "prompt",
                "final_answer",
            }
            missing = required_fields - row.keys()

            if missing:
                raise KeyError(
                    f"问题数据缺少字段 {sorted(missing)}"
                )

            # 保存原始训练集索引，便于复现实验。
            row["source_idx"] = source_idx
            items.append(row)

    if not items:
        raise RuntimeError(f"问题数据为空: {path}")

    if num_questions <= 0:
        raise ValueError("--num-questions 必须大于 0")

    if num_questions > len(items):
        print(
            f"[warn] 请求 {num_questions} 道题，"
            f"但数据集只有 {len(items)} 道，将使用全部"
        )
        num_questions = len(items)

    # 每个 outer step 使用不同但可复现的问题采样。
    question_seed = seed + grpo_step
    rng = random.Random(question_seed)
    selected = rng.sample(items, k=num_questions)

    print(
        f"[data] 从 {len(items)} 道题中抽取 "
        f"{len(selected)} 道"
    )
    print(
        f"[data] grpo_step={grpo_step}, "
        f"question_seed={question_seed}"
    )

    return selected


def default_output_path(grpo_step: int) -> Path:
    return (
        ROOT
        / "data"
        / f"grpo_rollout_step_{grpo_step:04d}.jsonl"
    )


def main() -> None:
    args = parse_args()

    if args.group_size <= 0:
        raise ValueError("--group-size 必须大于 0")

    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")

    questions = load_questions(
        path=Path(args.questions_path),
        num_questions=args.num_questions,
        seed=args.seed,
        grpo_step=args.grpo_step,
    )

    output_path = (
        Path(args.output)
        if args.output is not None
        else default_output_path(args.grpo_step)
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    from vllm import LLM, SamplingParams

    # 同一基础 seed 下，grpo_step 不同会产生不同 rollout。
    rollout_seed = args.seed + args.grpo_step

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        n=args.group_size,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=rollout_seed,
    )

    print(f"[init] 加载 old policy: {args.model}")
    print(
        f"[rollout] questions={len(questions)}, "
        f"group_size={args.group_size}, "
        f"expected={len(questions) * args.group_size}"
    )
    print(
        f"[sampling] temperature={args.temperature}, "
        f"top_p={args.top_p}, "
        f"min_tokens={args.min_tokens}, "
        f"max_tokens={args.max_tokens}, "
        f"seed={rollout_seed}"
    )

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    total_written = 0
    total_reward = 0.0

    cat_correct = 0
    cat_format_only = 0
    cat_unformatted = 0

    mixed_groups = 0
    all_correct_groups = 0
    all_wrong_groups = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for start in range(
            0,
            len(questions),
            args.batch_size,
        ):
            end = min(
                start + args.batch_size,
                len(questions),
            )
            batch_questions = questions[start:end]
            batch_prompts = [
                item["prompt"]
                for item in batch_questions
            ]

            print(
                f"[generate] 问题 {start + 1}~{end} "
                f"/ {len(questions)}，"
                f"每题 G={args.group_size}"
            )

            batch_outputs = llm.generate(
                batch_prompts,
                sampling_params,
            )

            if len(batch_outputs) != len(batch_questions):
                raise RuntimeError(
                    "vLLM 返回的问题数量与输入不一致："
                    f"{len(batch_outputs)} != "
                    f"{len(batch_questions)}"
                )

            for local_idx, request_output in enumerate(
                batch_outputs
            ):
                question_item = batch_questions[local_idx]

                completions = request_output.outputs
                if len(completions) != args.group_size:
                    raise RuntimeError(
                        "vLLM 返回的组大小不正确："
                        f"期望 {args.group_size}，"
                        f"实际 {len(completions)}"
                    )

                group_position = start + local_idx
                group_id = (
                    f"step_{args.grpo_step:04d}"
                    f"_group_{group_position:04d}"
                )

                prompt_token_ids = list(
                    request_output.prompt_token_ids
                )

                group_records = []
                group_rewards = []

                for sample_idx, completion in enumerate(
                    completions
                ):
                    response = completion.text
                    response_token_ids = list(
                        completion.token_ids
                    )

                    score = r1_zero_reward_fn(
                        response=response,
                        ground_truth=question_item[
                            "final_answer"
                        ],
                        fast=args.fast,
                    )

                    reward = float(score["reward"])
                    group_rewards.append(reward)
                    total_reward += reward

                    if (
                        score["format_reward"] == 1
                        and score["answer_reward"] == 1
                    ):
                        cat_correct += 1
                    elif score["format_reward"] == 1:
                        cat_format_only += 1
                    else:
                        cat_unformatted += 1

                    group_records.append(
                        {
                            "grpo_step": args.grpo_step,
                            "group_id": group_id,
                            "source_idx": question_item[
                                "source_idx"
                            ],
                            "sample_idx": sample_idx,
                            "question": question_item[
                                "question"
                            ],
                            "final_answer": question_item[
                                "final_answer"
                            ],
                            "prompt": question_item["prompt"],
                            "response": response,
                            "prompt_token_ids": (
                                prompt_token_ids
                            ),
                            "response_token_ids": (
                                response_token_ids
                            ),
                            "prompt_length": len(
                                prompt_token_ids
                            ),
                            "response_length": len(
                                response_token_ids
                            ),
                            "format_reward": float(
                                score["format_reward"]
                            ),
                            "answer_reward": float(
                                score["answer_reward"]
                            ),
                            "reward": reward,
                            "policy_model": args.model,
                        }
                    )

                reward_tensor = torch.tensor(
                    [group_rewards],
                    dtype=torch.float32,
                )

                advantage_tensor = (
                    compute_group_advantages(
                        rewards=reward_tensor,
                        normalize=(
                            args.normalize_advantages
                        ),
                        eps=args.advantage_eps,
                    )
                )

                group_advantages = (
                    advantage_tensor[0].tolist()
                )

                reward_sum = sum(group_rewards)
                if reward_sum == 0:
                    all_wrong_groups += 1
                elif reward_sum == args.group_size:
                    all_correct_groups += 1
                else:
                    mixed_groups += 1

                for record, advantage in zip(
                    group_records,
                    group_advantages,
                ):
                    record["advantage"] = float(advantage)
                    fout.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    total_written += 1

            fout.flush()

            print(
                f"[progress] 已写入 {total_written} / "
                f"{len(questions) * args.group_size}"
            )

    expected = len(questions) * args.group_size
    reward_mean = (
        total_reward / total_written
        if total_written > 0
        else 0.0
    )
    useful_group_fraction = (
        mixed_groups / len(questions)
        if questions
        else 0.0
    )

    print("[stats]")
    print(
        f"  正确且格式正确: {cat_correct}"
    )
    print(
        f"  格式正确但答案错误: {cat_format_only}"
    )
    print(
        f"  格式错误: {cat_unformatted}"
    )
    print(
        f"  all-correct groups: {all_correct_groups}"
    )
    print(
        f"  all-wrong groups: {all_wrong_groups}"
    )
    print(
        f"  mixed groups: {mixed_groups}"
    )
    print(
        f"  useful_group_fraction: "
        f"{useful_group_fraction:.4f}"
    )
    print(f"  reward_mean: {reward_mean:.4f}")
    print(
        f"[save] rollout: {output_path}"
    )
    print(
        f"[done] 实际写入 {total_written} / "
        f"期望 {expected}"
    )

    if total_written != expected:
        raise RuntimeError(
            "rollout 写入数量与期望不一致"
        )


if __name__ == "__main__":
    main()