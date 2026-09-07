from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grader.drgrpo_grader import extract_answer  

MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

PROMPT_FILE = ROOT / "prompts" / "r1_zero.prompt"
OUTPUT_FILE = ROOT / "data" / "math_sft_train.jsonl"

def load_math_train() -> Any:
    from datasets import concatenate_datasets, load_dataset

    parts = []
    for config in MATH_CONFIGS:
        print(f"[load] MATH/{config} train ...")
        parts.append(
            load_dataset("EleutherAI/hendrycks_math", config, split="train")
        )
    return concatenate_datasets(parts)


def build_samples() -> list[dict[str, str]]:
    template = PROMPT_FILE.read_text(encoding="utf-8").rstrip("\n")
    ds = load_math_train()

    samples: list[dict[str, str]] = []
    skipped_no_boxed = 0
    skipped_no_answer = 0

    for row in ds:
        solution = (row["solution"] or "").strip()

        if "\\boxed{" not in solution:
            skipped_no_boxed += 1
            continue

        final_answer = extract_answer(solution)

        if final_answer is None or not final_answer.strip():
            skipped_no_answer += 1
            continue
        final_answer = final_answer.strip()
        question = row["problem"]
        prompt = template.replace("{question}", question)
        response = f"{solution} </think> <answer>{final_answer}</answer>"
        samples.append(
            {
                "question": question,
                "final_answer": final_answer,
                "prompt": prompt,
                "response": response,
            }
        )

    print(f"[stats] 总样本数: {len(samples)}")
    print(f"[stats] 因缺少 \\boxed 跳过: {skipped_no_boxed}")
    print(f"[stats] 因提取答案失败跳过: {skipped_no_answer}")
    return samples

def main() -> None:
    samples = build_samples()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[save] 已写入: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()