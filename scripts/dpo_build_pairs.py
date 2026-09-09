from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Build DPO preference pairs")
    p.add_argument("--input", type=str, default=str(ROOT / "data" / "dpo_raw.jsonl"))
    p.add_argument("--output", type=str, default=str(ROOT / "data" / "dpo_pairs.jsonl"))
    p.add_argument("--questions-path", type=str,
                   default=str(ROOT / "data" / "math_sft_train.jsonl"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast", action="store_true", default=True)
    p.add_argument("--no-fast", dest="fast", action="store_false")
    return p.parse_args()


def load_ground_truth_map(path: Path) -> dict[str, str]:
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                mapping[row["question"]] = row["final_answer"]
    print(f"[gt] 标准答案数: {len(mapping)}")
    return mapping


def load_raw_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"[input] 原始回答数: {len(records)}")
    return records


def main():
    args = parse_args()
    from grader.drgrpo_grader import r1_zero_reward_fn

    gt_map = load_ground_truth_map(Path(args.questions_path))
    records = load_raw_records(Path(args.input))

    # 按题目分组：偏好对必须来自同一道题
    by_question: dict[str, list[dict]] = {}
    for rec in records:
        by_question.setdefault(rec["question"], []).append(rec)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = 0
    skipped_no_gt = 0
    skipped_no_chosen = 0
    skipped_no_rejected = 0

    with output_path.open("w", encoding="utf-8") as f:
        for question, q_records in by_question.items():
            gt = gt_map.get(question)
            if gt is None:
                skipped_no_gt += 1
                continue

            chosen_candidates = []
            rejected_candidates = []

            for rec in q_records:
                score = r1_zero_reward_fn(rec["response"], gt, fast=args.fast)

                if score["reward"] == 1:
                    chosen_candidates.append(rec)
                elif score["format_reward"] == 1:
                    # 格式对但答案错，才是合格的 rejected
                    rejected_candidates.append(rec)

            if not chosen_candidates:
                skipped_no_chosen += 1
                continue
            if not rejected_candidates:
                skipped_no_rejected += 1
                continue

            # 用 题目+种子 决定选哪条，保证可复现
            rng = random.Random(f"{args.seed}:{question}")
            chosen = rng.choice(chosen_candidates)
            rejected = rng.choice(rejected_candidates)

            f.write(json.dumps({
                "question": question,
                "prompt": chosen["prompt"],
                "chosen": chosen["response"],
                "rejected": rejected["response"],
                "final_answer": gt,
            }, ensure_ascii=False) + "\n")
            pairs += 1

    print(f"[pair] 成功配对: {pairs}")
    print(f"[pair] 跳过(无标准答案): {skipped_no_gt}")
    print(f"[pair] 跳过(无 chosen):  {skipped_no_chosen}")
    print(f"[pair] 跳过(无 rejected): {skipped_no_rejected}")
    print(f"[save] 写入: {output_path}")


if __name__ == "__main__":
    main()