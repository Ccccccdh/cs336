from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RSFT sampling")
    p.add_argument("--model", type=str,
                   default=str(ROOT / "models" / "sft_math"),
                   help="用于采样的当前策略模型")
    p.add_argument("--questions-path", type=str,
                   default=str(ROOT / "data" / "math_sft_train.jsonl"),
                   help="问题来源：复用 SFT 数据里的 prompt（已按 r1_zero 模板拼好）")
    p.add_argument("--num-questions", type=int, default=1000,
                   help="本轮抽取多少道题")
    p.add_argument("--G", type=int, default=8,
                   help="每个问题采样多少条回答")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--min-tokens", type=int, default=4,
                   help="避免生成空字符串")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32,
                   help="一次送给 vLLM 多少个问题（显存小就调小）")
    p.add_argument("--output", type=str,
                   default=str(ROOT / "data" / "rsft_round1_raw.jsonl"))
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return p.parse_args()


def load_questions(path: Path, num_questions: int, seed: int):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    if len(items) < num_questions:
        print(f"[warn] 问题总数 {len(items)} < 需要的 {num_questions}，将使用全部")
        num_questions = len(items)

    # 固定随机种子，保证同一批问题每次运行可复现
    selected = random.Random(seed).sample(items, k=num_questions)
    print(f"[data] 从 {len(items)} 道题中抽取 {len(selected)} 道")
    return selected


def main() -> None:
    args = parse_args()
    questions = load_questions(Path(args.questions_path),
                               args.num_questions,
                               args.seed)

    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        n=args.G,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )

    print(f"[init] 加载采样模型 {args.model} ...")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    prompts = [q["prompt"] for q in questions]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_expected = len(questions) * args.G
    written = 0

    with output_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(prompts), args.batch_size):
            end = min(start + args.batch_size, len(prompts))
            print(f"[generate] 问题 {start + 1}~{end} / {len(prompts)}，"
                  f"每问采样 G={args.G} ...")

            outputs = llm.generate(prompts[start:end], sampling_params)

            for i, output in enumerate(outputs):
                question = questions[start + i]

                for j, completion in enumerate(output.outputs):
                    record = {
                        "question": question["question"],
                        "prompt": question["prompt"],
                        "response": completion.text, 
                        "sample_idx": j,              
                        "model": args.model,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

            f.flush()
            print(f"[progress] 已写入 {written} / {total_expected} 条")

    print(f"[done] 原始采样保存到: {output_path}，共 {written} 条")
    print(f"期望值: {len(questions)} 题 × G={args.G} = {total_expected} 条")


if __name__ == "__main__":
    main()