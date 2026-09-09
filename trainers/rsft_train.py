from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="RSFT integrated trainer")
    p.add_argument("--model", type=str, default=str(ROOT / "models" / "sft_math"),
                   help="初始模型（第一轮从这里采样和继续训练）")
    p.add_argument("--questions-path", type=str,
                   default=str(ROOT / "data" / "math_sft_train.jsonl"),
                   help="问题库：复用 SFT 数据（含已拼好的 prompt 和 final_answer）")
    p.add_argument("--num-questions", type=int, default=1000,
                   help="每轮抽取多少道题")
    p.add_argument("--rounds", type=int, default=1,
                   help="RSFT 迭代轮数")
    p.add_argument("--G", type=int, default=8,
                   help="每个问题采样条数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast", action="store_true", default=True)
    p.add_argument("--no-fast", dest="fast", action="store_false")
    p.add_argument("--output-dir", type=str,
                   default=str(ROOT / "models" / "rsft_train"))

    # SFT 训练相关
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--max-seq-len", type=int, default=1536)
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--attn-impl", type=str, default="sdpa")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--log-every", type=int, default=10)

    # vLLM 采样相关
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--min-tokens", type=int, default=4)
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return p.parse_args()


def load_question_bank(path: Path):
    """读问题库；每轮从这里随机抽取 num_questions 道。"""
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    print(f"[bank] 问题库共 {len(items)} 道题")
    return items


def sample_round(model_path: str, questions, G: int, args) -> list[dict]:
    """第 1 步：vLLM 采样。用完必须 shutdown 释放显存。"""
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        n=G,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )

    print(f"[sample] 加载模型 {model_path} 采样 ...")
    llm = LLM(
        model=model_path,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    records = []
    prompts = [q["prompt"] for q in questions]
    try:
        for start in range(0, len(prompts), 32):  # 每批 32 题
            end = min(start + 32, len(prompts))
            outputs = llm.generate(prompts[start:end], sampling_params)
            for i, output in enumerate(outputs):
                question = questions[start + i]
                for j, completion in enumerate(output.outputs):
                    records.append({
                        "question": question["question"],
                        "prompt": question["prompt"],
                        "response": completion.text,
                    })
            print(f"[sample] 已采样 {len(records)} 条 ...")
    finally:
        # 关键：采样完立刻释放 vLLM，否则下一步 HF 训练会 OOM
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            shutdown()
        del llm
        torch.cuda.empty_cache()
        print("[sample] vLLM 已关闭，显存已释放")
    return records


def filter_round(records, bank_map, fast: bool):
    """第 2 步：筛选 reward==1 的样本，转成 SFT 数据格式。"""
    from grader.drgrpo_grader import r1_zero_reward_fn

    kept = []
    stats = {"correct": 0, "format_only": 0, "unformatted": 0, "unmatched": 0}

    for rec in records:
        ground_truth = bank_map.get(rec["question"])
        if ground_truth is None:
            stats["unmatched"] += 1
            continue

        score = r1_zero_reward_fn(rec["response"], ground_truth, fast=fast)

        if score["format_reward"] == 1 and score["answer_reward"] == 1:
            stats["correct"] += 1
        elif score["format_reward"] == 1:
            stats["format_only"] += 1
        else:
            stats["unformatted"] += 1

        if score["reward"] >= 1.0:
            kept.append({
                "question": rec["question"],
                "final_answer": ground_truth,
                "prompt": rec["prompt"],
                "response": rec["response"],
            })

    print(f"[filter] 正确 {stats['correct']} / 格式对答案错 {stats['format_only']} "
          f"/ 无格式 {stats['unformatted']} / 保留 {len(kept)}")
    return kept


def train_round(init_model_path: str, samples, args, round_idx: int) -> str:
    """第 3 步：用 SFT 逻辑训练一轮（复用 trainers/sft_train.py 的组件）。"""
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trainers.sft_train import SFTDataset, collate_fn, save_checkpoint
    from utils.loss import compute_sft_loss_and_entropy

    if not samples:
        raise RuntimeError("本轮没有筛选出任何正确样本")

    tokenizer = AutoTokenizer.from_pretrained(init_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        init_model_path,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    ).to("cuda")
    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(
        SFTDataset(samples),
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_seq_len),
    )

    total_updates = (len(samples) + args.micro_batch_size * args.grad_accum_steps - 1) \
                    // (args.micro_batch_size * args.grad_accum_steps)
    print(f"[train] round {round_idx}: {len(samples)} 条样本，"
          f"约 {total_updates} 次更新")

    running_loss, running_entropy, micros, update_idx = 0.0, 0.0, 0, 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        labels = batch["labels"].to("cuda")
        response_mask = batch["response_mask"].to("cuda")

        loss, entropy = compute_sft_loss_and_entropy(
            model, input_ids, attention_mask, labels, response_mask
        )
        (loss / args.grad_accum_steps).backward()

        running_loss += loss.item()
        running_entropy += entropy.item()
        micros += 1

        if micros == args.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            micros = 0
            update_idx += 1

            if update_idx % args.log_every == 0 or update_idx == total_updates:
                print(f"[epoch] update {update_idx}/{total_updates} "
                      f"loss={running_loss / args.grad_accum_steps:.4f} "
                      f"entropy={running_entropy / args.grad_accum_steps:.3f}")
                running_loss, running_entropy = 0.0, 0.0

    if micros > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        optimizer.zero_grad()

    save_path = Path(args.output_dir) / f"round_{round_idx}"
    save_checkpoint(model, tokenizer, save_path)
    return str(save_path)


def main():
    args = parse_args()
    bank = load_question_bank(Path(args.questions_path))
    bank_map = {q["question"]: q["final_answer"] for q in bank}

    current_model = args.model
    for round_idx in range(1, args.rounds + 1):
        print(f"\n===== RSFT round {round_idx} =====")
        # 每轮换 seed，避免采到几乎相同的输出
        questions = random.Random(args.seed + round_idx).sample(
            bank, k=min(args.num_questions, len(bank))
        )

        records = sample_round(current_model, questions, args.G, args)
        sft_samples = filter_round(records, bank_map, args.fast)

        if not sft_samples:
            print("[stop] 本轮没有正确样本，停止迭代")
            break

        current_model = train_round(current_model, sft_samples, args, round_idx)

    print(f"\n[done] 最终模型目录: {current_model}")
    print("用以下命令评测：")
    print(f"python baseline/zero_shot.py --dataset math "
          f"--model {current_model} --output-dir results/rsft_train_round{args.rounds}")


if __name__ == "__main__":
    main()