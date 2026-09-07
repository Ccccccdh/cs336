from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.loss import compute_sft_loss_and_entropy  
from utils.tokenization import tokenize_prompt_and_output  

 
# 1. Dataset：只管按索引返回一条样本的原始字符串
 

class SFTDataset(Dataset):
    """每一条样本是 {'prompt': ..., 'response': ..., 'question': ..., 'final_answer': ...}。"""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

 
# 2. collate_fn：把一个 batch 的字符串 tokenize 成 tensor
 

def collate_fn(batch, tokenizer, max_seq_len: int):
    """复用 utils/tokenization.py 的函数，并额外补上 attention_mask 和日志元信息。"""
    prompts = [item["prompt"] for item in batch]
    responses = [item["response"] for item in batch]

    # 返回的 response_mask 已经与 labels 对齐（True=要算 loss 的位置）
    input_ids, labels, response_mask = tokenize_prompt_and_output(
        prompts, responses, tokenizer, max_length=max_seq_len
    )

    # attention_mask：padding 位置是 0，其余是 1
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "response_mask": response_mask,
        # 下面这些只用于日志，不参与计算
        "questions": [item["question"] for item in batch],
        "final_answers": [item["final_answer"] for item in batch],
        "responses": [item["response"] for item in batch],
    }

 
# 3. 数据加载 + checkpoint 保存
 

def load_samples(data_path: Path, max_examples: int | None):
    if not data_path.exists():
        raise FileNotFoundError(
            f"找不到 {data_path}，请先运行: python data/build_sft_data.py"
        )
    samples = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if max_examples is not None:
        samples = samples[:max_examples]
    return samples


def save_checkpoint(model, tokenizer, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    print(f"[save] checkpoint 已保存: {path}")

 
# 4. 命令行参数

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--data-path", type=str, default=str(ROOT / "data" / "math_sft_train.jsonl"))
    p.add_argument("--max-examples", type=int, default=None, help="调试时只取前 N 条")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--micro-batch-size", type=int, default=2,
                   help="一次前向/反向的样本数，受显存限制")
    p.add_argument("--grad-accum-steps", type=int, default=16,
                   help="攒多少 micro-batch 更新一次；有效 batch = micro_batch * accum")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--output-dir", type=str, default=str(ROOT / "models" / "sft_math"))
    p.add_argument("--attn-impl", type=str, default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="显存不够时开启，用更多计算换显存")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10, help="每几次 update 打印一次日志")
    p.add_argument("--wandb", action="store_true", help="开启 wandb 记录")
    p.add_argument("--project", type=str, default="cs336-posttraining")
    return p.parse_args()


# 5. 主流程

def main():
    args = parse_args()

    # 可复现性：固定随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("SFT 需要 GPU 才能实际训练")

    # ---- 数据 ----
    samples = load_samples(Path(args.data_path), args.max_examples)
    dataset = SFTDataset(samples)
    print(f"[data] 训练样本数: {len(dataset)}")

    # ---- tokenizer 与模型 ----
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Qwen2.5 base 可能没有 pad_token；padding 不参与 loss，用 eos 顶替
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 新版本 transformers 用 dtype= 而不是 torch_dtype=
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    ).to(device)
    model.train()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("[opt] 已开启 gradient checkpointing")

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.project, config=vars(args))

    # ---- DataLoader：collate 里做 tokenize，避免把所有数据一次性编码占内存 ----
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_seq_len),
    )

    effective_batch = args.micro_batch_size * args.grad_accum_steps
    total_updates_per_epoch = (len(dataset) + effective_batch - 1) // effective_batch
    print(f"[train] 有效 batch size = {effective_batch}，"
          f"每 epoch 约 {total_updates_per_epoch} 次更新")

    global_update = 0  # 跨 epoch 的累计更新次数，用于 wandb 横轴

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        running_loss = 0.0
        running_entropy = 0.0
        micros_since_update = 0
        epoch_start = time.time()

        for step, batch in enumerate(loader):
            # 搬到 GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)

            # 算这一 micro-batch 的 loss 和熵
            loss, entropy = compute_sft_loss_and_entropy(
                model, input_ids, attention_mask, labels, response_mask
            )

            # 梯度累积：除以 accum 再 backward，等价于有效 batch 的平均梯度
            (loss / args.grad_accum_steps).backward()

            # 记录日志用的原始值
            running_loss += loss.item()
            running_entropy += entropy.item()
            micros_since_update += 1

            # 攒够 grad_accum_steps 个 micro-batch 才真正更新
            if micros_since_update == args.grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                micros_since_update = 0
                global_update += 1

                avg_loss = running_loss / args.grad_accum_steps
                avg_entropy = running_entropy / args.grad_accum_steps
                running_loss = 0.0
                running_entropy = 0.0

                if global_update % args.log_every == 0 or global_update == total_updates_per_epoch:
                    elapsed = time.time() - epoch_start
                    msg = (f"[epoch {epoch}] update {global_update}/{total_updates_per_epoch} "
                           f"loss={avg_loss:.4f} entropy={avg_entropy:.3f} "
                           f"time={elapsed:.1f}s")
                    print(msg)

                    if run is not None:
                        run.log({
                            "epoch": epoch,
                            "train/update": global_update,
                            "train/loss": avg_loss,
                            "train/entropy": avg_entropy,
                            "train/lr": optimizer.param_groups[0]["lr"],
                        })
                        # 记录少量训练样本，方便回溯模型当时在学什么
                        table = run.Table(columns=["question", "final_answer", "target_response"])
                        for q, a, r in zip(batch["questions"][:3],
                                           batch["final_answers"][:3],
                                           batch["responses"][:3]):
                            table.add_data(q, a, r[:300])
                        run.log({"train/samples": table})

        # epoch 末尾：不足 accum 的残余梯度也要用掉
        if micros_since_update > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_update += 1

        # 每个 epoch 保存一份完整 checkpoint（模型 + tokenizer）
        save_checkpoint(model, tokenizer, Path(args.output_dir) / f"epoch_{epoch}")
        print(f"[epoch {epoch}] 完成，耗时 {time.time() - epoch_start:.1f}s")

    save_checkpoint(model, tokenizer, Path(args.output_dir))

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()