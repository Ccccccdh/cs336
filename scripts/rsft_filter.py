from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RSFT filtering")
    p.add_argument("--input", type=str,
                   default=str(ROOT / "data" / "rsft_round1_raw.jsonl"),
                   help="采样脚本的输出")
    p.add_argument("--output", type=str,
                   default=str(ROOT / "data" / "rsft_round1_sft.jsonl"),
                   help="筛选后的 SFT 训练数据")
    p.add_argument("--questions-path", type=str,
                   default=str(ROOT / "data" / "math_sft_train.jsonl"),
                   help="用于取 ground truth（final_answer）")
    p.add_argument("--min-reward", type=float, default=1.0,
                   help="保留 reward >= 该值的样本；1.0 表示格式和答案都正确")
    p.add_argument("--fast", action="store_true", default=True)
    p.add_argument("--no-fast", dest="fast", action="store_false")
    return p.parse_args()


def load_ground_truth_map(path: Path) -> dict[str, str]:
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            mapping[row["question"]] = row["final_answer"]
    print(f"[gt] 加载标准答案 {len(mapping)} 条")
    return mapping


def main() -> None:
    args = parse_args()

    from grader.drgrpo_grader import r1_zero_reward_fn

    gt_map = load_ground_truth_map(Path(args.questions_path))

    records = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"[input] 原始采样共 {len(records)} 条")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    cat_correct = 0      
    cat_format_only = 0  
    cat_unformatted = 0 
    unmatched = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for rec in records:
            ground_truth = gt_map.get(rec["question"])
            if ground_truth is None:
                unmatched += 1
                continue

            score = r1_zero_reward_fn(
                rec["response"],
                ground_truth,
                fast=args.fast,
            )
            if score["format_reward"] == 1 and score["answer_reward"] == 1:
                cat_correct += 1
            elif score["format_reward"] == 1:
                cat_format_only += 1
            else:
                cat_unformatted += 1

            if score["reward"] >= args.min_reward:
                sft_record = {
                    "question": rec["question"],
                    "final_answer": ground_truth,
                    "prompt": rec["prompt"],
                    "response": rec["response"],
                }
                fout.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
                kept += 1

    print(f"[filter] 保留 {kept} / {len(records)} 条")
    print(f"[filter] (1,1) 格式对+答案对: {cat_correct}")
    print(f"[filter] (1,0) 格式对答案错: {cat_format_only}")
    print(f"[filter] (0,0) 格式错:     {cat_unformatted}")
    print(f"[filter] 找不到标准答案:     {unmatched}")
    print(f"[save] 筛选结果写入: {output_path}")

    if kept == 0:
        raise RuntimeError(
            "没有筛选出任何正确样本。建议增大 G 或增加 num-questions 后重新采样。"
        )


if __name__ == "__main__":
    main()