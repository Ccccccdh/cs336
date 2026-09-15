"""按任务书生成并验证 Self-Play 数学问题。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grader.drgrpo_grader import r1_zero_reward_fn  # noqa: E402
from utils.self_play import (  # noqa: E402
    extract_model_answer,
    format_solve_prompt,
    normalize_problem,
    parse_problem_and_answer,
    validate_generated_problem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Problem+Answer pairs and verify them with the policy"
    )
    parser.add_argument("--model", default="models/grpo_fast")
    parser.add_argument("--questions-path", default="data/math_sft_train.jsonl")
    parser.add_argument(
        "--problem-prompt-file",
        default="prompts/self_play_problem_gen.prompt",
    )
    parser.add_argument("--solve-prompt-file", default="prompts/r1_zero.prompt")
    parser.add_argument("--output", default="data/self_play_generated_pilot.jsonl")
    parser.add_argument(
        "--raw-output",
        default="data/self_play_generated_pilot_raw.jsonl",
    )
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--candidates-per-seed", type=int, default=2)
    parser.add_argument("--problem-temperature", type=float, default=0.9)
    parser.add_argument("--problem-top-p", type=float, default=0.95)
    parser.add_argument("--problem-max-tokens", type=int, default=768)
    parser.add_argument(
        "--fallback-answer-temperature",
        type=float,
        default=0.2,
        help="Problem 可解析但 Answer 缺失时，补答案所用温度",
    )
    parser.add_argument("--verify-temperature", type=float, default=0.7)
    parser.add_argument("--verify-top-p", type=float, default=0.95)
    parser.add_argument(
        "--verify-samples",
        type=int,
        default=2,
        help="独立验证同一自产答案的解答数；全部正确才接收",
    )
    parser.add_argument("--solve-max-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_question(row: dict[str, Any]) -> str | None:
    for key in ("question", "problem", "query"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return None


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
    if args.verify_samples < 1:
        raise ValueError("--verify-samples 必须至少为 1")
    from vllm import LLM, SamplingParams

    rows = load_jsonl(Path(args.questions_path))
    seeds: list[dict[str, Any]] = []
    for source_idx, row in enumerate(rows):
        question = get_question(row)
        if question:
            seeds.append({"source_idx": source_idx, "question": question})
    if not seeds:
        raise RuntimeError("questions 文件中没有 question/problem/query 字段")

    rng = random.Random(args.seed)
    selected = rng.sample(seeds, min(args.num_seeds, len(seeds)))
    problem_template = Path(args.problem_prompt_file).read_text(encoding="utf-8")
    solve_template = Path(args.solve_prompt_file).read_text(encoding="utf-8")
    proposer_prompts = [
        problem_template.replace("{seed_problem}", item["question"])
        for item in selected
    ]

    print(f"[model] loading: {args.model}")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=args.seed,
    )

    proposer_params = SamplingParams(
        n=args.candidates_per_seed,
        temperature=args.problem_temperature,
        top_p=args.problem_top_p,
        min_tokens=16,
        max_tokens=args.problem_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    raw_records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    seen_problems: set[str] = set()
    reasons: Counter[str] = Counter()
    problem_parsed = 0
    proposer_answer_parsed = 0

    print(
        f"[propose] seeds={len(selected)}, "
        f"candidates/seed={args.candidates_per_seed}"
    )
    for start, prompt_batch in batched(proposer_prompts, args.batch_size):
        request_outputs = llm.generate(prompt_batch, proposer_params)
        for local_idx, request_output in enumerate(request_outputs):
            seed_item = selected[start + local_idx]
            for sample_idx, completion in enumerate(request_output.outputs):
                response = completion.text
                problem, answer = parse_problem_and_answer(response)
                record: dict[str, Any] = {
                    "source_idx": seed_item["source_idx"],
                    "sample_idx": sample_idx,
                    "seed_problem": seed_item["question"],
                    "proposer_prompt": prompt_batch[local_idx],
                    "proposer_response": response,
                    "finish_reason": getattr(completion, "finish_reason", None),
                    "problem": problem,
                    "generated_answer": answer,
                    "answer_source": "proposer" if answer is not None else None,
                    "rejection_reason": None,
                }
                raw_index = len(raw_records)
                raw_records.append(record)

                if problem is None:
                    record["rejection_reason"] = "problem_parse_failed"
                    reasons["problem_parse_failed"] += 1
                    continue
                problem_parsed += 1
                if answer is not None:
                    proposer_answer_parsed += 1

                rejection = validate_generated_problem(
                    problem=problem,
                    seed_problem=seed_item["question"],
                )
                if rejection:
                    record["rejection_reason"] = rejection
                    reasons[rejection] += 1
                    continue

                normalized = normalize_problem(problem)
                if normalized in seen_problems:
                    record["rejection_reason"] = "duplicate_generated_problem"
                    reasons["duplicate_generated_problem"] += 1
                    continue
                seen_problems.add(normalized)
                candidates.append(
                    {
                        "raw_index": raw_index,
                        "source_idx": seed_item["source_idx"],
                        "seed_problem": seed_item["question"],
                        "problem": problem,
                        "answer": answer,
                        "answer_source": "proposer" if answer is not None else None,
                        "proposer_response": response,
                    }
                )
        done = min(start + len(prompt_batch), len(selected))
        print(f"[propose] processed {done}/{len(selected)} seeds")

    # 兼容当前 solver checkpoint：若只成功输出 Problem，则再调用一次同一模型补答案。
    missing_indices = [i for i, item in enumerate(candidates) if item["answer"] is None]
    fallback_created = 0
    if missing_indices:
        fallback_prompts = [
            format_solve_prompt(solve_template, candidates[i]["problem"])
            for i in missing_indices
        ]
        fallback_params = SamplingParams(
            n=1,
            temperature=args.fallback_answer_temperature,
            top_p=1.0,
            min_tokens=4,
            max_tokens=args.solve_max_tokens,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
        print(f"[fallback] missing proposer answers: {len(missing_indices)}")
        for start, prompt_batch in batched(fallback_prompts, args.batch_size):
            request_outputs = llm.generate(prompt_batch, fallback_params)
            for local_idx, request_output in enumerate(request_outputs):
                candidate_idx = missing_indices[start + local_idx]
                candidate = candidates[candidate_idx]
                response = request_output.outputs[0].text
                answer = extract_model_answer(response)
                raw = raw_records[candidate["raw_index"]]
                raw["fallback_answer_response"] = response
                raw["fallback_answer"] = answer
                if answer is None:
                    raw["rejection_reason"] = "fallback_answer_parse_failed"
                    reasons["fallback_answer_parse_failed"] += 1
                    continue
                candidate["answer"] = answer
                candidate["answer_source"] = "fallback_solver"
                raw["generated_answer"] = answer
                raw["answer_source"] = "fallback_solver"
                fallback_created += 1
            done = min(start + len(prompt_batch), len(fallback_prompts))
            print(f"[fallback] processed {done}/{len(fallback_prompts)}")

    verifiable = [item for item in candidates if item["answer"] is not None]
    verify_prompts = [
        format_solve_prompt(solve_template, item["problem"]) for item in verifiable
    ]
    verify_params = SamplingParams(
        n=args.verify_samples,
        temperature=args.verify_temperature,
        top_p=args.verify_top_p,
        min_tokens=4,
        max_tokens=args.solve_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    accepted: list[dict[str, Any]] = []
    print(f"[verify] problem+answer pairs: {len(verifiable)}")
    for start, prompt_batch in batched(verify_prompts, args.batch_size):
        request_outputs = llm.generate(prompt_batch, verify_params)
        for local_idx, request_output in enumerate(request_outputs):
            candidate = verifiable[start + local_idx]
            responses = [completion.text for completion in request_output.outputs]
            reward_infos = [
                r1_zero_reward_fn(response, candidate["answer"], fast=True)
                for response in responses
            ]
            raw = raw_records[candidate["raw_index"]]
            raw["verification_prompt"] = prompt_batch[local_idx]
            raw["verification_responses"] = responses
            raw["verification_rewards"] = reward_infos

            if not all(info["reward"] == 1.0 for info in reward_infos):
                reason = (
                    "verification_format_failed"
                    if any(info["format_reward"] != 1.0 for info in reward_infos)
                    else "verification_answer_incorrect"
                )
                raw["rejection_reason"] = reason
                reasons[reason] += 1
                continue

            raw["rejection_reason"] = None
            accepted.append(
                {
                    "self_play_id": f"self_play_{len(accepted):06d}",
                    "source_idx": candidate["source_idx"],
                    "seed_problem": candidate["seed_problem"],
                    "question": candidate["problem"],
                    "problem": candidate["problem"],
                    "ground_truth": candidate["answer"],
                    "answer": candidate["answer"],
                    "answer_source": candidate["answer_source"],
                    "proposer_response": candidate["proposer_response"],
                    "verification_response": responses[0],
                    "verification_reward": reward_infos[0],
                    "verification_responses": responses,
                    "verification_rewards": reward_infos,
                }
            )
        done = min(start + len(prompt_batch), len(verifiable))
        print(f"[verify] processed {done}/{len(verifiable)}")

    output_path = Path(args.output)
    raw_output_path = Path(args.raw_output)
    write_jsonl(output_path, accepted)
    write_jsonl(raw_output_path, raw_records)

    generated = len(raw_records)
    stats = {
        "generated": generated,
        "problem_parsed": problem_parsed,
        "proposer_answer_parsed": proposer_answer_parsed,
        "static_valid": len(candidates),
        "fallback_answer_created": fallback_created,
        "verification_attempted": len(verifiable),
        "verification_samples_per_problem": args.verify_samples,
        "verification_responses": len(verifiable) * args.verify_samples,
        "accepted": len(accepted),
        "problem_parse_rate": problem_parsed / generated if generated else 0.0,
        "acceptance_rate": len(accepted) / generated if generated else 0.0,
        "rejections": dict(sorted(reasons.items())),
    }
    print("[stats]")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[save] accepted: {output_path}")
    print(f"[save] raw: {raw_output_path}")
    if not accepted:
        print("[warning] 本轮没有通过验证的样本；raw 输出已保留，不抛出异常。")
    sys.stdout.flush()

    del llm
    gc.collect()
    os._exit(0)


if __name__ == "__main__":
    main()
