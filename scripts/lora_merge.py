"""把手写 LoRA Adapter 合并进基础模型并保存为普通 HF 模型。

为减少合并误差，默认在 CPU/FP32 中计算 delta_W，合并完成后再转换成
BF16 保存。输出模型不再包含 LoRALinear，可直接由 transformers 或 vLLM 加载。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lora import (  # noqa: E402
    LoRALinear,
    load_lora_adapter,
    lora_module_names,
    merge_lora_weights,
    parameter_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge manual LoRA into base model")
    parser.add_argument("--base-model", default="models/self_play_exp2_step3")
    parser.add_argument("--adapter-dir", default="models/phyx_lora_r4")
    parser.add_argument("--output-dir", default="models/phyx_lora_r4_merged")
    parser.add_argument(
        "--merge-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="FP32 合并误差更小，但需要约 6GB CPU 内存",
    )
    parser.add_argument(
        "--save-dtype",
        choices=["float32", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument("--allow-base-mismatch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def ensure_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"输出目录非空: {path}；如确认覆盖，请显式指定 --overwrite"
        )
    path.mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(adapter_dir)
    ensure_output_path(output_dir, args.overwrite)

    merge_dtype = resolve_dtype(args.merge_dtype)
    save_dtype = resolve_dtype(args.save_dtype)
    print(
        f"[model] loading base on CPU: {args.base_model} "
        f"(merge_dtype={args.merge_dtype})"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=merge_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    base_parameter_count = count_parameters(model)

    config = load_lora_adapter(model, adapter_dir, strict=True)
    configured_base = config.get("base_model")
    if (
        configured_base
        and str(configured_base) != str(args.base_model)
        and not args.allow_base_mismatch
    ):
        raise ValueError(
            "Adapter 记录的基础模型与 --base-model 不一致:\n"
            f"adapter={configured_base}\nargument={args.base_model}\n"
            "若你已经人工确认兼容，可指定 --allow-base-mismatch。"
        )

    before_report = parameter_report(model)
    print("[adapter]")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print("[parameters before merge]")
    print(json.dumps(before_report, ensure_ascii=False, indent=2))

    merged_names = merge_lora_weights(model, unload=True)
    if len(merged_names) != len(config.get("modules", merged_names)):
        raise RuntimeError(
            f"合并模块数异常: merged={len(merged_names)}, "
            f"configured={len(config.get('modules', []))}"
        )
    remaining = lora_module_names(model)
    if remaining or any(isinstance(module, LoRALinear) for module in model.modules()):
        raise RuntimeError(f"合并后仍存在 LoRA 模块: {remaining}")
    merged_parameter_count = count_parameters(model)
    if merged_parameter_count != base_parameter_count:
        raise RuntimeError(
            "卸载 LoRA 后参数量没有恢复为基础模型参数量: "
            f"base={base_parameter_count}, merged={merged_parameter_count}"
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = True
    if save_dtype != merge_dtype:
        print(f"[model] converting merged model to {args.save_dtype}")
        model.to(dtype=save_dtype)

    print(f"[save] full merged model: {output_dir}")
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer_source = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    tokenizer.save_pretrained(output_dir)

    merge_info: dict[str, Any] = {
        "base_model": args.base_model,
        "adapter_dir": str(adapter_dir),
        "output_dir": str(output_dir),
        "merge_dtype": args.merge_dtype,
        "save_dtype": args.save_dtype,
        "merged_modules": len(merged_names),
        "base_parameters": base_parameter_count,
        "merged_parameters": merged_parameter_count,
        "adapter_config": config,
        "directly_loadable_by_vllm": True,
    }
    (output_dir / "merge_info.json").write_text(
        json.dumps(merge_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[verify] LoRA modules remaining: 0")
    print(f"[verify] parameters: {merged_parameter_count:,}")
    print(f"[done] merged model: {output_dir}")


if __name__ == "__main__":
    main()
