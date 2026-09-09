from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dpo import compute_sequence_log_probs  # noqa: E402
from utils.tokenization import tokenize_prompt_and_output  # noqa: E402


# 1. Dataset + collate：一条样本 = (prompt, chosen, rejected)

class DPOPairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = dict(self.pairs[idx])
        item["idx"] = idx  # 用于对齐 ref 缓存
        return item


def collate_fn(batch, tokenizer, max_seq_len: int):
    prompts = [b["prompt"] for b in batch]
    chosens = [b["chosen"] for b in batch]
    rejected = [b["rejected"] for b in batch]

    # chosen / rejected 分别 tokenize，各自得到 [B, T] 的四个张量
    chosen_ids, chosen_labels, chosen_mask = tokenize_prompt_and_output(
        prompts, chosens, tokenizer, max_length=max_seq_len
    )
    rej_ids, rej_labels, rej_mask = tokenize_prompt_and_output(
        prompts, rejected, tokenizer, max_length=max_seq_len
    )

    return {
        "indices": [b["idx"] for b in batch],
        "chosen_input_ids": chosen_ids,
        "chosen_attention_mask": (chosen_ids != tokenizer.pad_token_id).long(),
        "chosen_labels": chosen_labels,
        "chosen_response_mask": chosen_mask,
        "rej_input_ids": rej_ids,
        "rej_attention_mask": (rej_ids != tokenizer.pad_token_id).long(),
        "rej_labels": rej_labels,
        "rej_response_mask": rej_mask,
    }


# 2. reference log_prob 预计算 + 缓存

def compute_ref_cache(model_path, dataset, cache_path, tokenizer, max_seq_len):
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"[cache] 读取已有 ref 缓存: {cache_path} ({len(data)} 条)")
        return data

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    ref_model.eval()

    loader = DataLoader(
        dataset,
        batch_size=8,         
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_seq_len),
    )

    cache = []
    print(f"[ref] 开始预计算 {len(dataset)} 条偏好对的 ref log_prob ...")
    with torch.no_grad():
        for batch in loader:
            chosen_logp = compute_sequence_log_probs(
                ref_model,
                batch["chosen_input_ids"].to("cuda"),
                batch["chosen_attention_mask"].to("cuda"),
                batch["chosen_labels"].to("cuda"),
                batch["chosen_response_mask"].to("cuda"),
            )
            rej_logp = compute_sequence_log_probs(
                ref_model,
                batch["rej_input_ids"].to("cuda"),
                batch["rej_attention_mask"].to("cuda"),
                batch["rej_labels"].to("cuda"),
                batch["rej_response_mask"].to("cuda"),
            )
            for w, l in zip(chosen_logp.tolist(), rej_logp.tolist()):
                cache.append([w, l])

    del ref_model
    torch.cuda.empty_cache()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[ref] ref 缓存已保存: {cache_path}")
    return cache


# 3. 参数

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default=str(ROOT / "models" / "rsft_round2_full"),
                   help="policy 初始模型 = reference 模型（RSFT 后的自己）")
    p.add_argument("--data-path", type=str,
                   default=str(ROOT / "data" / "dpo_pairs_pilot.jsonl"))
    p.add_argument("--cache-path", type=str,
                   default=str(ROOT / "data" / "dpo_ref_cache_pilot.json"))
    p.add_argument("--output-dir", type=str,
                   default=str(ROOT / "models" / "dpo_pilot"))
    p.add_argument("--beta", type=float, default=0.2,
                   help="KL 系数，任务书建议 0.1~0.5")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--micro-batch-size", type=int, default=1,
                   help="一次前向处理的偏好对数量")
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--max-seq-len", type=int, default=1536)
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


# 4. 主流程

def main():
    args = parse_args()

    # 读取偏好数据
    pairs = []
    with Path(args.data_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    print(f"[data] 偏好对数量: {len(pairs)}")
    if not pairs:
        raise RuntimeError("没有偏好数据，请先运行 dpo_build_pairs.py")

    dataset = DPOPairDataset(pairs)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cache = compute_ref_cache(
        args.model, dataset, Path(args.cache_path), tokenizer, args.max_seq_len
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("[opt] 已开启 gradient checkpointing")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_seq_len),
    )

    total_updates = (len(pairs) + args.micro_batch_size * args.grad_accum_steps - 1) \
                    // (args.micro_batch_size * args.grad_accum_steps)
    print(f"[train] 每 epoch 约 {total_updates} 次更新")

    running_loss = 0.0
    running_acc = 0.0
    micros = 0
    update_idx = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(loader):
            indices = batch["indices"]

            # 从缓存取出本 batch 对应的 ref log_prob
            ref_w = torch.tensor([cache[i][0] for i in indices],
                                 dtype=torch.float32, device="cuda")
            ref_l = torch.tensor([cache[i][1] for i in indices],
                                 dtype=torch.float32, device="cuda")

            # policy 对 chosen / rejected 的 log_prob（带梯度）
            logp_w = compute_sequence_log_probs(
                model,
                batch["chosen_input_ids"].to("cuda"),
                batch["chosen_attention_mask"].to("cuda"),
                batch["chosen_labels"].to("cuda"),
                batch["chosen_response_mask"].to("cuda"),
            )
            logp_l = compute_sequence_log_probs(
                model,
                batch["rej_input_ids"].to("cuda"),
                batch["rej_attention_mask"].to("cuda"),
                batch["rej_labels"].to("cuda"),
                batch["rej_response_mask"].to("cuda"),
            )

            # DPO loss：让 chosen 的相对提升 > rejected 的相对提升
            logits_diff = (logp_w - ref_w) - (logp_l - ref_l)
            loss = -F.logsigmoid(args.beta * logits_diff).mean()

            # 训练中"选对"的比例（chosen 得分高于 rejected）
            acc = (logits_diff > 0).float().mean().item()

            (loss / args.grad_accum_steps).backward()

            running_loss += loss.item()
            running_acc += acc
            micros += 1

            if micros == args.grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                micros = 0
                update_idx += 1

                # 每次更新后都计算并清零，而不是只在打印时清零
                avg_loss = running_loss / args.grad_accum_steps
                avg_acc = running_acc / args.grad_accum_steps
                running_loss = 0.0
                running_acc = 0.0

                if update_idx % args.log_every == 0 or update_idx == total_updates:
                    print(f"[epoch {epoch}] update {update_idx}/{total_updates} "
                          f"loss={avg_loss:.4f} acc={avg_acc:.4f} "
                          f"time={time.time()-t0:.1f}s")

        if micros > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            optimizer.zero_grad()

        # 每个 epoch 保存
        save_dir = Path(args.output_dir) / f"epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        print(f"[save] checkpoint: {save_dir}")

    final_dir = Path(args.output_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[done] 最终模型: {final_dir}")


if __name__ == "__main__":
    main()