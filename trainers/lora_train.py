"""不依赖 PEFT 的 Qwen2.5-Math FFN LoRA 监督微调。

基础模型全部冻结，只向 gate_proj、up_proj、down_proj 注入并训练低秩矩阵。
训练 JSONL 需要包含 prompt 和 response；prompt 的 token label 全部设为 -100，
因此损失只来自 response。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lora import (  # noqa: E402
    LoRALinear,
    inject_lora,
    load_lora_adapter,
    parameter_report,
    save_lora_adapter,
)


TRAINING_INSTRUCTION = (
    "Solve the physics problem. Explain the reasoning, then put only the "
    "correct option letter inside the <answer> tags."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train manual FFN LoRA on PhyX")
    parser.add_argument("--model", default="models/self_play_exp2_step3")
    parser.add_argument("--train-data", default="data/phyx_lora/train.jsonl")
    parser.add_argument(
        "--validation-data", default="data/phyx_lora/validation.jsonl"
    )
    parser.add_argument("--output-dir", default="models/phyx_lora_r4")
    parser.add_argument(
        "--objective",
        choices=["reasoning", "answer_only"],
        default="reasoning",
        help="answer_only 与 A-D likelihood 评测直接对齐",
    )
    parser.add_argument(
        "--balance-answers",
        action="store_true",
        help="按 A-D 标签逆频率采样，仅建议 answer_only 使用",
    )

    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="FP32 adapter 更稳定；基础模型仍保持 BF16",
    )
    parser.add_argument(
        "--target-suffixes",
        nargs="+",
        default=["gate_proj", "up_proj", "down_proj"],
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-validation-examples", type=int, default=None)

    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-impl",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--resume-adapter",
        default=None,
        help="从已有 adapter 继续；其 rank/alpha/dropout 配置优先",
    )
    parser.add_argument(
        "--resume-state",
        default=None,
        help="恢复 optimizer/scheduler/global_step 状态",
    )
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def read_jsonl(path: Path, max_examples: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
            if not row.get("prompt") or not row.get("response"):
                raise ValueError(f"{path}:{line_number} 缺少 prompt/response")
            rows.append(row)
            if max_examples is not None and len(rows) >= max_examples:
                break
    if not rows:
        raise RuntimeError(f"{path} 没有有效样本")
    return rows


def strip_training_instruction(question: str) -> str:
    question = question.strip()
    if question.endswith(TRAINING_INSTRUCTION):
        question = question[: -len(TRAINING_INSTRUCTION)].rstrip()
    return question


def answer_only_prompt(question: str) -> str:
    return (
        "A conversation between User and Assistant. The User asks a "
        "multiple-choice physics question.\n"
        f"User: {strip_training_instruction(question)}\n\n"
        "Assistant: The correct option is\n<answer>"
    )


def apply_objective(
    rows: list[dict[str, Any]], objective: str
) -> list[dict[str, Any]]:
    if objective == "reasoning":
        return rows
    converted: list[dict[str, Any]] = []
    for row in rows:
        answer = str(row.get("final_answer") or "").strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"样本 id={row.get('id')} 的 final_answer 不是 A-D")
        item = dict(row)
        item["prompt"] = answer_only_prompt(str(row.get("question") or ""))
        # likelihood 评测只比较 <answer> 后的下一个 A-D token；这里也只监督
        # 这个字母，避免固定的 </answer>/EOS token 稀释分类梯度。
        item["response"] = answer
        converted.append(item)
    return converted


def answer_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(row.get("final_answer") or "").strip().upper()
                for row in rows
            ).items()
        )
    )


def middle_truncate(tokens: list[int], limit: int) -> list[int]:
    """同时保留文本开头和结尾；题目选项与 answer 标签通常位于结尾。"""

    if len(tokens) <= limit:
        return tokens
    if limit <= 0:
        return []
    head = limit // 2
    return tokens[:head] + tokens[-(limit - head) :]


class CompletionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_seq_len: int,
        append_eos: bool = True,
    ) -> None:
        self.examples: list[dict[str, Any]] = []
        self.stats = {
            "source_examples": len(rows),
            "truncated_examples": 0,
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0,
        }
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise RuntimeError("tokenizer.eos_token_id 为空")

        for row in rows:
            prompt_ids = tokenizer.encode(
                str(row["prompt"]), add_special_tokens=False
            )
            response_ids = tokenizer.encode(
                str(row["response"]), add_special_tokens=False
            )
            if append_eos and (not response_ids or response_ids[-1] != eos_id):
                response_ids.append(eos_id)

            original_length = len(prompt_ids) + len(response_ids)
            if original_length > max_seq_len:
                self.stats["truncated_examples"] += 1
                # 先压缩 response，但保留开头步骤和末尾 <answer>；至少给题面留一半。
                response_limit = min(len(response_ids), max_seq_len // 2)
                response_ids = middle_truncate(response_ids, response_limit)
                prompt_limit = max_seq_len - len(response_ids)
                prompt_ids = middle_truncate(prompt_ids, prompt_limit)

            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + list(response_ids)
            if len(input_ids) > max_seq_len or len(input_ids) != len(labels):
                raise RuntimeError("token 截断逻辑错误")
            if not any(label != -100 for label in labels):
                raise RuntimeError("样本没有可训练 response token")

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "id": str(row.get("id", "")),
                }
            )
            self.stats["prompt_tokens"] += len(prompt_ids)
            self.stats["response_tokens"] += len(response_ids)
            self.stats["total_tokens"] += len(input_ids)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class CompletionCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(example["input_ids"]) for example in examples)
        input_ids = torch.full(
            (len(examples), max_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        for index, example in enumerate(examples):
            length = len(example["input_ids"])
            input_ids[index, :length] = torch.tensor(
                example["input_ids"], dtype=torch.long
            )
            labels[index, :length] = torch.tensor(example["labels"], dtype=torch.long)
            attention_mask[index, :length] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def cast_lora_parameters(model: nn.Module, dtype: torch.dtype) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.data = module.lora_A.data.to(dtype=dtype)
            module.lora_B.data = module.lora_B.data.to(dtype=dtype)


def build_loader(
    dataset: CompletionDataset,
    collator: CompletionCollator,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    sample_weights: list[float] | None = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    if sample_weights is not None:
        if len(sample_weights) != len(dataset):
            raise ValueError("sample_weights 与 dataset 长度不一致")
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator if shuffle and sampler is None else None,
    )


def move_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.to("cuda", non_blocking=True)
        for key, value in batch.items()
    }


@torch.inference_mode()
def validation_loss(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    weighted_loss = 0.0
    supervised_tokens = 0
    for batch in loader:
        batch = move_batch(batch)
        count = int((batch["labels"] != -100).sum().item())
        output = model(**batch)
        weighted_loss += float(output.loss.item()) * count
        supervised_tokens += count
    model.train()
    return weighted_loss / supervised_tokens if supervised_tokens else float("nan")


def save_training_state(
    path: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        },
        path,
    )


def load_training_state(
    path: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> tuple[int, int]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"])
    if "cuda_rng_state_all" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    return int(state.get("epoch", 0)), int(state.get("global_step", 0))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA 训练需要 CUDA GPU")
    if args.rank <= 0 or args.epochs <= 0:
        raise ValueError("rank 和 epochs 必须为正数")
    if args.micro_batch_size <= 0 or args.grad_accum_steps <= 0:
        raise ValueError("batch size 和 gradient accumulation 必须为正数")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup-ratio 必须在 [0, 1) 内")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[tokenizer] {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = read_jsonl(Path(args.train_data), args.max_train_examples)
    validation_rows = read_jsonl(
        Path(args.validation_data), args.max_validation_examples
    )
    print(f"[objective] {args.objective}")
    print(f"[labels] train={answer_distribution(train_rows)}")
    print(f"[labels] validation={answer_distribution(validation_rows)}")
    train_rows = apply_objective(train_rows, args.objective)
    validation_rows = apply_objective(validation_rows, args.objective)
    append_eos = args.objective != "answer_only"
    train_dataset = CompletionDataset(
        train_rows, tokenizer, args.max_seq_len, append_eos=append_eos
    )
    validation_dataset = CompletionDataset(
        validation_rows,
        tokenizer,
        args.max_seq_len,
        append_eos=append_eos,
    )
    print("[data] train:")
    print(json.dumps(train_dataset.stats, ensure_ascii=False, indent=2))
    print("[data] validation:")
    print(json.dumps(validation_dataset.stats, ensure_ascii=False, indent=2))

    print(f"[model] loading frozen base: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_impl,
    )
    if args.resume_adapter:
        adapter_config = load_lora_adapter(model, args.resume_adapter)
        print(f"[lora] resumed adapter: {args.resume_adapter}")
        print(json.dumps(adapter_config, ensure_ascii=False, indent=2))
    else:
        names = inject_lora(
            model,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.lora_dropout,
            target_suffixes=args.target_suffixes,
            freeze_base=True,
        )
        print(f"[lora] injected {len(names)} modules")

    adapter_dtype = (
        torch.float32 if args.lora_dtype == "float32" else torch.bfloat16
    )
    cast_lora_parameters(model, adapter_dtype)
    report = parameter_report(model)
    print("[parameters]")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["trainable_parameters"] != report["lora_parameters"]:
        raise RuntimeError("存在非 LoRA 可训练参数")

    model.to("cuda")
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        enable_inputs = getattr(model, "enable_input_require_grads", None)
        if callable(enable_inputs):
            enable_inputs()
        print("[memory] gradient checkpointing enabled")

    collator = CompletionCollator(tokenizer.pad_token_id)
    sample_weights = None
    if args.balance_answers:
        if args.objective != "answer_only":
            raise ValueError("--balance-answers 只允许配合 --objective answer_only")
        counts = Counter(str(row["final_answer"]).strip().upper() for row in train_rows)
        sample_weights = [
            1.0 / counts[str(row["final_answer"]).strip().upper()]
            for row in train_rows
        ]
        print("[sampling] inverse-frequency answer balancing enabled")
    train_loader = build_loader(
        train_dataset,
        collator,
        args.micro_batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
        sample_weights=sample_weights,
    )
    validation_loader = build_loader(
        validation_dataset,
        collator,
        args.micro_batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
        sample_weights=None,
    )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = round(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    start_epoch = 0
    global_step = 0
    if args.resume_state:
        start_epoch, global_step = load_training_state(
            Path(args.resume_state), optimizer, scheduler
        )
        print(
            f"[resume] state={args.resume_state}, "
            f"completed_epochs={start_epoch}, global_step={global_step}"
        )
    if start_epoch >= args.epochs:
        raise ValueError("resume state 中已完成的 epoch 不小于 --epochs")

    initial_validation_loss = validation_loss(model, validation_loader)
    print(f"[validation] before training loss={initial_validation_loss:.6f}")
    history: list[dict[str, Any]] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    training_started = time.time()

    for epoch_index in range(start_epoch, args.epochs):
        epoch_number = epoch_index + 1
        running_loss = 0.0
        running_micro_steps = 0
        epoch_started = time.time()
        for micro_step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch)
            output = model(**batch)
            raw_loss = output.loss
            (raw_loss / args.grad_accum_steps).backward()
            running_loss += float(raw_loss.detach().item())
            running_micro_steps += 1

            should_update = (
                micro_step % args.grad_accum_steps == 0
                or micro_step == len(train_loader)
            )
            if not should_update:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, args.clip_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.log_every == 0 or micro_step == len(train_loader):
                average_loss = running_loss / max(1, running_micro_steps)
                print(
                    f"[train] epoch={epoch_number}/{args.epochs} "
                    f"micro={micro_step}/{len(train_loader)} "
                    f"update={global_step}/{total_updates} "
                    f"loss={average_loss:.6f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} "
                    f"grad_norm={float(grad_norm):.4f}"
                )
                running_loss = 0.0
                running_micro_steps = 0

        val_loss = validation_loss(model, validation_loader)
        epoch_record = {
            "epoch": epoch_number,
            "global_step": global_step,
            "validation_loss": val_loss,
            "epoch_seconds": round(time.time() - epoch_started, 2),
        }
        history.append(epoch_record)
        print(f"[validation] epoch={epoch_number} loss={val_loss:.6f}")

        if args.save_every_epoch:
            checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch_number}"
            save_lora_adapter(
                model,
                checkpoint_dir,
                base_model=args.model,
                extra_config={
                    "completed_epochs": epoch_number,
                    "global_step": global_step,
                    "training_args": vars(args),
                },
            )
            tokenizer.save_pretrained(checkpoint_dir)
            save_training_state(
                checkpoint_dir / "trainer_state.pt",
                optimizer,
                scheduler,
                epoch_number,
                global_step,
            )
            print(f"[save] checkpoint: {checkpoint_dir}")

    adapter_config = save_lora_adapter(
        model,
        output_dir,
        base_model=args.model,
        extra_config={
            "completed_epochs": args.epochs,
            "global_step": global_step,
            "training_args": vars(args),
        },
    )
    tokenizer.save_pretrained(output_dir)
    save_training_state(
        output_dir / "trainer_state.pt",
        optimizer,
        scheduler,
        args.epochs,
        global_step,
    )
    summary = {
        "base_model": args.model,
        "train_data": args.train_data,
        "validation_data": args.validation_data,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": history[-1]["validation_loss"],
        "parameter_report": report,
        "adapter_config": adapter_config,
        "history": history,
        "total_training_seconds": round(time.time() - training_started, 2),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[save] adapter: {output_dir}")
    print(f"[save] summary: {output_dir / 'training_summary.json'}")
    print("[done] LoRA training complete")


if __name__ == "__main__":
    main()
