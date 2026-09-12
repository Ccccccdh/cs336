from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.grpo import (
    compute_grpo_clip_loss,
    compute_token_log_probs,
)


class GRPORolloutDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.old_log_probs = [None] * len(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = dict(self.records[idx])
        item["idx"] = idx
        return item


def load_records(path: Path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise RuntimeError(f"rollout 文件为空: {path}")

    reward_mean = sum(
        float(x["reward"]) for x in records
    ) / len(records)

    active_count = sum(
        abs(float(x["advantage"])) > 1e-8
        for x in records
    )

    print(f"[data] rollout 样本数: {len(records)}")
    print(f"[data] 平均 reward: {reward_mean:.4f}")
    print(
        f"[data] 非零 advantage: "
        f"{active_count}/{len(records)}"
    )

    return records


def collate_fn(
    batch,
    tokenizer,
    max_seq_len,
    old_cache=None,
):
    input_rows = []
    attention_rows = []
    label_rows = []
    response_mask_rows = []
    old_logp_rows = []

    for item in batch:
        prompt_ids = list(item["prompt_token_ids"])
        response_ids = list(item["response_token_ids"])

        if len(prompt_ids) >= max_seq_len:
            raise ValueError(
                f"prompt 长度 {len(prompt_ids)} "
                f">= max_seq_len {max_seq_len}"
            )

        # 保留完整 prompt，只截断 response 尾部。
        response_ids = response_ids[
            : max_seq_len - len(prompt_ids)
        ]

        if not response_ids:
            raise ValueError("response token 为空")

        input_ids = prompt_ids + response_ids
        seq_len = len(input_ids)

        labels = [-100] * seq_len
        response_mask = [False] * seq_len

        # prompt 最后一个位置预测第一个 response token。
        start = len(prompt_ids) - 1
        end = start + len(response_ids)

        labels[start:end] = response_ids
        response_mask[start:end] = [
            True
        ] * len(response_ids)

        input_rows.append(input_ids)
        attention_rows.append([1] * seq_len)
        label_rows.append(labels)
        response_mask_rows.append(response_mask)

        if old_cache is not None:
            cached = old_cache[item["idx"]]

            if cached is None:
                raise RuntimeError(
                    "old log-prob 尚未计算"
                )

            cached = cached[: len(response_ids)]

            if len(cached) != len(response_ids):
                raise RuntimeError(
                    "old log-prob 与 response 长度不一致"
                )

            old_row = [0.0] * seq_len
            old_row[start:end] = cached.tolist()
            old_logp_rows.append(old_row)

    batch_len = max(len(row) for row in input_rows)
    pad_id = tokenizer.pad_token_id

    def pad(rows, value):
        return [
            row + [value] * (batch_len - len(row))
            for row in rows
        ]

    result = {
        "indices": [item["idx"] for item in batch],
        "input_ids": torch.tensor(
            pad(input_rows, pad_id),
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            pad(attention_rows, 0),
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            pad(label_rows, -100),
            dtype=torch.long,
        ),
        "response_mask": torch.tensor(
            pad(response_mask_rows, False),
            dtype=torch.bool,
        ),
        "advantages": torch.tensor(
            [
                float(item["advantage"])
                for item in batch
            ],
            dtype=torch.float32,
        ),
    }

    if old_cache is not None:
        result["old_log_probs"] = torch.tensor(
            pad(old_logp_rows, 0.0),
            dtype=torch.float32,
        )

    return result


def compute_old_log_prob_cache(
    model,
    dataset,
    tokenizer,
    max_seq_len,
    batch_size,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch,
            tokenizer,
            max_seq_len,
        ),
    )

    cache = [None] * len(dataset)
    model.eval()

    print("[old] 开始计算 old policy token log-prob")

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch[
                "attention_mask"
            ].cuda()
            labels = batch["labels"].cuda()
            response_mask = batch[
                "response_mask"
            ].cuda()

            token_log_probs = compute_token_log_probs(
                model,
                input_ids,
                attention_mask,
                labels,
            )

            for row_idx, dataset_idx in enumerate(
                batch["indices"]
            ):
                values = token_log_probs[row_idx][
                    response_mask[row_idx]
                ]

                cache[dataset_idx] = (
                    values.float().cpu()
                )

            if step % 10 == 0 or step == len(loader):
                print(
                    f"[old] batch {step}/{len(loader)}"
                )

    print("[old] old log-prob 缓存完成")
    return cache


def parse_args():
    p = argparse.ArgumentParser(
        description="Train on one GRPO rollout batch"
    )

    p.add_argument(
        "--model",
        type=str,
        default=str(ROOT / "models" / "dpo_beta01"),
    )
    p.add_argument(
        "--rollout-path",
        type=str,
        default=str(
            ROOT / "data" / "grpo_rollout_smoke.jsonl"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(
            ROOT / "models" / "grpo_current"
        ),
    )
    p.add_argument(
        "--optimizer-state-path",
        type=str,
        default=None,
        help="跨 GRPO outer step 保存 AdamW 状态",
    )

    p.add_argument("--epochs", type=int, default=2)
    p.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
    )
    p.add_argument(
        "--old-logprob-batch-size",
        type=int,
        default=1,
    )
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=1536,
    )
    p.add_argument(
        "--clip-eps",
        type=float,
        default=0.2,
    )
    p.add_argument(
        "--clip-norm",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=1)

    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("GRPO 训练需要 CUDA GPU")

    records = load_records(Path(args.rollout_path))
    dataset = GRPORolloutDataset(records)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[init] 加载 policy: {args.model}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()

    # 此时模型还没有更新，因此就是 old policy。
    dataset.old_log_probs = compute_old_log_prob_cache(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        batch_size=args.old_logprob_batch_size,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("[opt] 已开启 gradient checkpointing")

    model.config.use_cache = False
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # 恢复前一个 GRPO outer step 的 AdamW 状态。
    if args.optimizer_state_path is not None:
        optimizer_path = Path(
            args.optimizer_state_path
        )

        if optimizer_path.exists():
            state = torch.load(
                optimizer_path,
                map_location="cuda",
            )
            optimizer.load_state_dict(state)

            # 命令行中的学习率优先。
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate

            print(
                f"[opt] 已恢复 optimizer: "
                f"{optimizer_path}"
            )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: collate_fn(
            batch,
            tokenizer,
            args.max_seq_len,
            dataset.old_log_probs,
        ),
    )

    updates_per_epoch = math.ceil(
        len(loader) / args.grad_accum_steps
    )
    total_updates = (
        updates_per_epoch * args.epochs
    )

    print(
        f"[train] epochs={args.epochs}, "
        f"micro_batch={args.micro_batch_size}, "
        f"grad_accum={args.grad_accum_steps}"
    )
    print(
        f"[train] 总计约 {total_updates} 次更新"
    )

    optimizer.zero_grad()
    update_idx = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        micros = 0
        running_loss = 0.0
        running_ratio = 0.0
        running_clip = 0.0
        running_kl = 0.0

        for batch in loader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch[
                "attention_mask"
            ].cuda()
            labels = batch["labels"].cuda()
            response_mask = batch[
                "response_mask"
            ].cuda()
            old_log_probs = batch[
                "old_log_probs"
            ].cuda()
            advantages = batch[
                "advantages"
            ].cuda()

            current_log_probs = compute_token_log_probs(
                model,
                input_ids,
                attention_mask,
                labels,
            )

            loss, metrics = compute_grpo_clip_loss(
                current_log_probs,
                old_log_probs,
                advantages,
                response_mask,
                clip_eps=args.clip_eps,
            )

            (
                loss / args.grad_accum_steps
            ).backward()

            micros += 1
            running_loss += loss.item()
            running_ratio += metrics["ratio_mean"]
            running_clip += metrics["clip_fraction"]
            running_kl += metrics["approx_kl"]

            if micros == args.grad_accum_steps:
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.clip_norm,
                    )
                )

                optimizer.step()
                optimizer.zero_grad()
                update_idx += 1

                if (
                    update_idx % args.log_every == 0
                    or update_idx == total_updates
                ):
                    print(
                        f"[epoch {epoch}] "
                        f"update {update_idx}/{total_updates} "
                        f"loss={running_loss / micros:.6f} "
                        f"ratio={running_ratio / micros:.4f} "
                        f"clip={running_clip / micros:.4f} "
                        f"kl={running_kl / micros:.6f} "
                        f"grad={float(grad_norm):.4f} "
                        f"time={time.time()-start_time:.1f}s"
                    )

                micros = 0
                running_loss = 0.0
                running_ratio = 0.0
                running_clip = 0.0
                running_kl = 0.0

        # 正式配置下 256 条样本可以整除累积步数，
        # 这里仍兼容最后不足一次累积的情况。
        if micros > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.clip_norm,
            )
            optimizer.step()
            optimizer.zero_grad()
            update_idx += 1

            print(
                f"[epoch {epoch}] "
                f"update {update_idx}/{total_updates} "
                f"loss={running_loss / micros:.6f} "
                f"ratio={running_ratio / micros:.4f} "
                f"clip={running_clip / micros:.4f} "
                f"kl={running_kl / micros:.6f} "
                f"grad={float(grad_norm):.4f}"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.config.use_cache = True
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    model.config.use_cache = False

    print(f"[save] policy: {output_dir}")

    if args.optimizer_state_path is not None:
        optimizer_path = Path(
            args.optimizer_state_path
        )
        optimizer_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        torch.save(
            optimizer.state_dict(),
            optimizer_path,
        )
        print(
            f"[save] optimizer: {optimizer_path}"
        )

    print("[done] GRPO rollout batch 训练完成")


if __name__ == "__main__":
    main()