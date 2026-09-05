"""
Zero-shot baseline：用 Qwen2.5-Math-1.5B 直接回答数学题，
并用 grader/drgrpo_grader.py 的 r1_zero_reward_fn 打分。

流程：
  1. 加载评测集（MATH / GSM8K / 本地 jsonl），统一成 question + ground_truth；
  2. 用 prompts/r1_zero.prompt 拼 prompt；
  3. vLLM 批量生成（temperature=1.0, top_p=1.0, max_tokens=1024,
     stop=["</answer>"], include_stop_str_in_output=True）；
  4. 调 r1_zero_reward_fn 打分；
  5. 输出逐样本 jsonl + 汇总 metrics.json。

注意：Qwen2.5-Math-1.5B 是 base 模型，不要对 prompt 使用 apply_chat_template。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# 保证无论从哪里执行都能 import 到项目根目录下的 grader 包
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot MATH/GSM8K baseline evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="本地模型路径或 HuggingFace 模型 ID，默认 Qwen/Qwen2.5-Math-1.5B",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["math", "gsm8k", "jsonl"],
        default="math",
        help="评测数据集：math=hendrycks_math test，gsm8k=openai/gsm8k test，jsonl=本地文件",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="当 --dataset jsonl 时指定本地 jsonl；"
        "也支持用本地文件覆盖 math/gsm8k 的在线加载（字段需含 question/problem 与 answer/solution）",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="最多评测多少条（默认全部），调试时建议先用 20~200 条",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=str(ROOT / "prompts" / "r1_zero.prompt"),
        help="提示词模板文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "results" / "baseline"),
        help="结果输出目录",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="每批送给 vLLM 生成的样本数",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="r1_zero_reward_fn 的 fast=True（更快，偶尔误判）；"
        "正式报告前可加 --no-fast 用 fast=False 再跑一遍",
    )
    parser.add_argument("--no-fast", dest="fast", action="store_false")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="vLLM 权重精度：auto/bfloat16/float16 等",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM 使用的 GPU 数量（单卡默认 1）",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="vLLM 最多使用多少比例的显存",
    )
    parser.add_argument("--seed", type=int, default=None, help="vLLM 采样随机种子（可选）")
    return parser.parse_args()


# ---------------------------------------------------------------- 数据加载

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _gsm8k_answer(text: str) -> str:
    """GSM8K 的 answer 形如 '...\n#### 8'，取 #### 后面的数字。"""
    if "####" in text:
        return text.rsplit("####", 1)[1].strip()
    return text.strip()


def _math_ground_truths(solution: str) -> list[str]:
    """MATH 的 solution 是完整解答，答案通常在 \\boxed{...} 里。

    这里返回候选列表：[完整 solution] + [提取出的 \\boxed 内容]（若有）。
    r1_zero_reward_fn 支持 ground_truth 为 list，任一命中即算正确。
    """
    from grader.drgrpo_grader import extract_answer  # 复用官方提取逻辑

    candidates = [solution.strip()]
    if "\\boxed" in solution:
        boxed = extract_answer(solution)
        if boxed and boxed not in candidates:
            candidates.append(boxed)
    # 去空去重，保持顺序
    return list(dict.fromkeys(c for c in candidates if c))


def _row_to_items(row: dict[str, Any]) -> dict[str, Any] | None:
    """把一行（jsonl/HF sample）标准化成 question + ground_truth。"""
    question = row.get("question") or row.get("problem")
    if not question:
        return None
    answer = row.get("answer")
    solution = row.get("solution")
    if solution:
        ground_truth = _math_ground_truths(str(solution))
    elif answer is not None:
        ground_truth = _gsm8k_answer(str(answer))
    else:
        return None
    return {"question": str(question), "ground_truth": ground_truth}


def load_items(dataset: str, data_path: str | None, max_examples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]]
    if data_path:
        rows = _load_jsonl(Path(data_path))
    elif dataset == "gsm8k":
        from datasets import load_dataset

        rows = list(load_dataset("openai/gsm8k", "main", split="test"))
    elif dataset == "math":
        from datasets import load_dataset

        rows = list(load_dataset("EleutherAI/hendrycks_math", "all", split="test"))
    else:
        raise ValueError("--dataset jsonl 时必须提供 --data-path")

    items = []
    for row in rows:
        item = _row_to_items(row)
        if item is not None:
            items.append(item)
        if max_examples is not None and len(items) >= max_examples:
            break
    if not items:
        raise RuntimeError("没有读到任何有效样本，请检查数据字段是否为 question/problem + answer/solution")
    return items


# ---------------------------------------------------------------- 生成与打分

def format_prompt(template: str, question: str) -> str:
    return template.replace("{question}", question)


def generate_and_score(
    items: list[dict[str, Any]],
    template: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], float]:
    from vllm import LLM, SamplingParams

    from grader.drgrpo_grader import r1_zero_reward_fn

    prompts = [format_prompt(template, item["question"]) for item in items]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )
    print(f"[init] 加载模型 {args.model} ...")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / f"zero_shot_{args.dataset}_samples.jsonl"

    rows: list[dict[str, Any]] = []
    total_time = 0.0
    with samples_path.open("w", encoding="utf-8") as fout:
        for start in range(0, len(prompts), args.batch_size):
            end = min(start + args.batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            print(f"[generate] 第 {start + 1}~{end} / {len(prompts)} 条，生成中 ...")

            t0 = time.time()
            outputs = llm.generate(batch_prompts, sampling_params)
            elapsed = time.time() - t0
            total_time += elapsed

            for i, output in enumerate(outputs):
                idx = start + i
                response = output.outputs[0].text
                score = r1_zero_reward_fn(
                    response,
                    items[idx]["ground_truth"],
                    fast=args.fast,
                )
                record = {
                    "index": idx,
                    "question": items[idx]["question"],
                    "ground_truth": items[idx]["ground_truth"],
                    "prompt": batch_prompts[i],
                    "response": response,
                    "response_len_tokens": len(output.outputs[0].token_ids),
                    **score,
                }
                rows.append(record)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    return rows, total_time


def _write_metrics(
    rows: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
    extra: dict[str, Any],
) -> None:
    n = len(rows)
    fmt_ok = sum(1 for r in rows if r["format_reward"] == 1)
    ans_ok = sum(1 for r in rows if r["answer_reward"] == 1)
    reward_ok = sum(1 for r in rows if r["reward"] == 1)
    cat_correct = sum(1 for r in rows if r["format_reward"] == 1 and r["answer_reward"] == 1)
    cat_format_only = sum(1 for r in rows if r["format_reward"] == 1 and r["answer_reward"] == 0)
    cat_unformatted = sum(1 for r in rows if r["format_reward"] == 0 and r["answer_reward"] == 0)

    metrics = {
        "dataset": args.dataset,
        "data_path": args.data_path,
        "model": args.model,
        "num_examples": n,
        "answer_accuracy": round(ans_ok / n, 6) if n else 0.0,
        "format_rate": round(fmt_ok / n, 6) if n else 0.0,
        "reward_mean": round(reward_ok / n, 6) if n else 0.0,
        "category_correct_1_1": cat_correct,
        "category_format_only_1_0": cat_format_only,
        "category_unformatted_0_0": cat_unformatted,
        "fast": args.fast,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
        **extra,
    }
    metrics_path = output_dir / f"zero_shot_{args.dataset}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] 结果写入: {output_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    template = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if "{question}" not in template:
        raise ValueError(f"提示词模板缺少 {{question}} 占位符: {args.prompt_file}")

    print(f"[data] 加载数据集: {args.dataset} (data_path={args.data_path})")
    items = load_items(args.dataset, args.data_path, args.max_examples)
    print(f"[data] 共 {len(items)} 条有效样本")

    rows, total_seconds = generate_and_score(items, template, args)
    _write_metrics(
        rows,
        Path(args.output_dir),
        args,
        {"total_generation_seconds": round(total_seconds, 2)},
    )


if __name__ == "__main__":
    main()
