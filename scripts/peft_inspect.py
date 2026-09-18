"""PEFT 第一步：检查 Qwen FFN 结构与 PhyX 数据结构。

本脚本只做只读检查，不修改模型，也不执行训练。
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any


TARGET_SUFFIXES = ("gate_proj", "up_proj", "down_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 LoRA 目标层和 PhyX 字段")
    parser.add_argument("--model", default="models/self_play_exp2_step3")
    parser.add_argument("--dataset", default="Cloudriver/PhyX")
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="默认优先选择名为 default 的 config，否则选择第一个 config",
    )
    parser.add_argument(
        "--dataset-split",
        default=None,
        help="用于读取示例；默认优先 test，否则选择第一个 split",
    )
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="只检查数据集，不加载模型（用于快速比较不同 dataset config）",
    )
    parser.add_argument("--sample-chars", type=int, default=300)
    parser.add_argument("--skip-dataset-sample", action="store_true")
    parser.add_argument("--output", default="results/peft_inspection.json")
    return parser.parse_args()


def jsonable_config_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable_config_value(item) for item in value]
    return str(value)


def inspect_model(model_name: str, ranks: list[int]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    print(f"[model] loading on CPU: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    config = model.config
    config_fields = (
        "model_type",
        "architectures",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "torch_dtype",
    )
    config_summary = {
        field: jsonable_config_value(getattr(config, field, None))
        for field in config_fields
    }

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    targets: list[dict[str, Any]] = []
    non_linear_targets: list[str] = []
    counts_by_suffix = {suffix: 0 for suffix in TARGET_SUFFIXES}

    print("[model] LoRA target modules:")
    for name, module in model.named_modules():
        matched_suffix = next(
            (suffix for suffix in TARGET_SUFFIXES if name.endswith(suffix)),
            None,
        )
        if matched_suffix is None:
            continue
        if not isinstance(module, torch.nn.Linear):
            non_linear_targets.append(name)
            print(f"  [warning] {name}: {type(module).__name__}, not nn.Linear")
            continue

        counts_by_suffix[matched_suffix] += 1
        target = {
            "name": name,
            "kind": matched_suffix,
            "in_features": module.in_features,
            "out_features": module.out_features,
            "has_bias": module.bias is not None,
            "weight_shape": list(module.weight.shape),
        }
        targets.append(target)
        print(
            f"  {name}: Linear({module.in_features}, {module.out_features}), "
            f"weight={tuple(module.weight.shape)}, bias={module.bias is not None}"
        )

    rank_estimates: dict[str, dict[str, float | int]] = {}
    for rank in ranks:
        if rank <= 0:
            raise ValueError("--ranks 中的值必须为正整数")
        trainable = sum(
            rank * (target["in_features"] + target["out_features"])
            for target in targets
        )
        rank_estimates[str(rank)] = {
            "trainable_parameters": trainable,
            "percentage_of_base": 100.0 * trainable / total_parameters,
            "bf16_adapter_megabytes": trainable * 2 / (1024**2),
        }

    expected_layers = getattr(config, "num_hidden_layers", None)
    expected_target_count = (
        3 * expected_layers if isinstance(expected_layers, int) else None
    )
    print("[model] summary")
    print(f"  architecture: {config_summary['architectures']}")
    print(f"  layers: {expected_layers}")
    print(f"  hidden/intermediate: {config_summary['hidden_size']}/{config_summary['intermediate_size']}")
    print(f"  total parameters: {total_parameters:,}")
    print(f"  target modules: {len(targets)} (expected {expected_target_count})")
    print(f"  target counts: {counts_by_suffix}")
    for rank, estimate in rank_estimates.items():
        print(
            f"  rank={rank}: {estimate['trainable_parameters']:,} trainable "
            f"({estimate['percentage_of_base']:.4f}%), "
            f"adapter bf16≈{estimate['bf16_adapter_megabytes']:.2f} MiB"
        )

    summary = {
        "name_or_path": model_name,
        "config": config_summary,
        "total_parameters": total_parameters,
        "target_suffixes": list(TARGET_SUFFIXES),
        "target_count": len(targets),
        "expected_target_count": expected_target_count,
        "counts_by_suffix": counts_by_suffix,
        "non_linear_targets": non_linear_targets,
        "targets": targets,
        "rank_estimates": rank_estimates,
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def safe_preview(value: Any, max_chars: int) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "..."
    if isinstance(value, bytes):
        return f"<bytes: {len(value)}>"
    if isinstance(value, list):
        return [safe_preview(item, max_chars) for item in value[:8]]
    if isinstance(value, dict):
        return {
            str(key): safe_preview(item, max_chars)
            for key, item in value.items()
            if key != "bytes"
        }
    if hasattr(value, "size") and hasattr(value, "mode"):
        return f"<image mode={value.mode} size={value.size}>"
    text = str(value)
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def choose_name(requested: str | None, available: list[str], preferred: str) -> str:
    if requested is not None:
        if requested not in available:
            raise ValueError(f"请求的名称 {requested!r} 不在可选项 {available} 中")
        return requested
    if preferred in available:
        return preferred
    if not available:
        raise RuntimeError("没有发现可用项")
    return available[0]


def inspect_dataset(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import (
        Image,
        get_dataset_config_names,
        get_dataset_split_names,
        load_dataset,
        load_dataset_builder,
    )

    print(f"[dataset] inspecting: {args.dataset}")
    configs = get_dataset_config_names(args.dataset)
    selected_config = choose_name(args.dataset_config, configs, "default")
    splits = get_dataset_split_names(args.dataset, selected_config)
    selected_split = choose_name(args.dataset_split, splits, "test")
    builder = load_dataset_builder(args.dataset, selected_config)

    features = {
        name: str(feature) for name, feature in (builder.info.features or {}).items()
    }
    split_sizes = {
        name: info.num_examples for name, info in (builder.info.splits or {}).items()
    }
    print(f"  configs: {configs}")
    print(f"  selected config: {selected_config}")
    print(f"  splits: {splits}")
    print(f"  split sizes: {split_sizes}")
    print("  features:")
    for name, feature in features.items():
        print(f"    {name}: {feature}")

    sample_preview: dict[str, Any] | None = None
    resolved_fields: dict[str, str | None] = {}
    if not args.skip_dataset_sample:
        stream = load_dataset(
            args.dataset,
            selected_config,
            split=selected_split,
            streaming=True,
        )
        if "image" in stream.features and isinstance(stream.features["image"], Image):
            stream = stream.cast_column("image", Image(decode=False))
        sample = next(iter(stream))
        sample_preview = {
            key: safe_preview(value, args.sample_chars)
            for key, value in sample.items()
        }
        aliases = {
            "id": ("id", "index"),
            "question": ("question",),
            "description": (
                "question_description_simplified",
                "question_simply",
                "question_description",
            ),
            "options": ("options",),
            "answer": ("answer",),
            "image_caption": ("image_caption",),
            "category": ("category",),
            "subfield": ("subfield",),
            "reasoning_type": ("reasoning_type",),
        }
        for logical_name, candidates in aliases.items():
            resolved_fields[logical_name] = next(
                (candidate for candidate in candidates if candidate in sample),
                None,
            )
        print("  resolved fields:")
        for logical_name, actual_name in resolved_fields.items():
            print(f"    {logical_name}: {actual_name}")
        print("  sample preview:")
        print(json.dumps(sample_preview, ensure_ascii=False, indent=2))

    return {
        "name": args.dataset,
        "configs": configs,
        "selected_config": selected_config,
        "splits": splits,
        "selected_split": selected_split,
        "split_sizes": split_sizes,
        "features": features,
        "resolved_fields": resolved_fields,
        "sample": sample_preview,
        "note": (
            "PhyX 官方提供的是评测 split；后续将创建固定、分层且互不重叠的课程实验划分。"
        ),
    }


def main() -> None:
    args = parse_args()
    summary: dict[str, Any] = {}
    if args.skip_model:
        print("[model] skipped")
    else:
        summary["model"] = inspect_model(args.model, args.ranks)
    try:
        summary["dataset"] = inspect_dataset(args)
    except Exception as exc:  # 仍保存已经完成的模型检查，便于定位网络或数据配置问题。
        summary["dataset_error"] = f"{type(exc).__name__}: {exc}"
        print(f"[dataset] inspection failed: {summary['dataset_error']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[save] inspection summary: {output_path}")


if __name__ == "__main__":
    main()
