"""不依赖 PEFT 的 LoRA 实现。

约定 PyTorch Linear 权重 W 的形状为 [out_features, in_features]，并使用
delta_W = B @ A，其中 A=[rank, in_features]、B=[out_features, rank]。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F


ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.pt"
DEFAULT_TARGET_SUFFIXES = ("gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """在冻结的 ``nn.Linear`` 旁路加入低秩更新。"""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer 必须是 torch.nn.Linear")
        if rank <= 0:
            raise ValueError("rank 必须为正整数")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 内")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.merged = False

        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

        factory_kwargs = {
            "device": base_layer.weight.device,
            "dtype": base_layer.weight.dtype,
        }
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, base_layer.in_features, **factory_kwargs)
        )
        self.lora_B = nn.Parameter(
            torch.empty(base_layer.out_features, self.rank, **factory_kwargs)
        )
        self.reset_lora_parameters()

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def reset_lora_parameters(self) -> None:
        # B=0 保证初始 delta_W=0，注入 LoRA 不会改变基础模型输出。
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(inputs)
        if self.merged:
            return result

        lora_inputs = self.lora_dropout(inputs).to(self.lora_A.dtype)
        update = F.linear(F.linear(lora_inputs, self.lora_A), self.lora_B)
        return result + update.to(result.dtype) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        self.base_layer.weight.add_(
            self.delta_weight().to(
                device=self.base_layer.weight.device,
                dtype=self.base_layer.weight.dtype,
            )
        )
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        self.base_layer.weight.sub_(
            self.delta_weight().to(
                device=self.base_layer.weight.device,
                dtype=self.base_layer.weight.dtype,
            )
        )
        self.merged = False

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling}, "
            f"merged={self.merged}"
        )


def _parent_and_child(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    if "." not in module_name:
        return model, module_name
    parent_name, child_name = module_name.rsplit(".", 1)
    return model.get_submodule(parent_name), child_name


def freeze_model(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def inject_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_suffixes: Iterable[str] = DEFAULT_TARGET_SUFFIXES,
    freeze_base: bool = True,
) -> list[str]:
    """将目标 ``nn.Linear`` 原地替换为 ``LoRALinear``。"""

    suffixes = tuple(target_suffixes)
    if not suffixes:
        raise ValueError("target_suffixes 不能为空")
    if freeze_base:
        freeze_model(model)

    targets: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not name.endswith(suffixes):
            continue
        if isinstance(module, LoRALinear):
            raise RuntimeError(f"目标层已经注入 LoRA: {name}")
        if not isinstance(module, nn.Linear):
            raise TypeError(f"目标层不是 nn.Linear: {name} ({type(module).__name__})")
        targets.append((name, module))

    if not targets:
        raise RuntimeError(f"没有找到后缀为 {suffixes} 的 nn.Linear")

    injected_names: list[str] = []
    for name, base_layer in targets:
        parent, child_name = _parent_and_child(model, name)
        setattr(
            parent,
            child_name,
            LoRALinear(
                base_layer=base_layer,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )
        injected_names.append(name)
    return injected_names


def lora_module_names(model: nn.Module) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, LoRALinear)]


def lora_parameter_names(model: nn.Module) -> list[str]:
    return [
        name
        for name, _ in model.named_parameters()
        if name.endswith(".lora_A") or name.endswith(".lora_B")
    ]


def parameter_report(model: nn.Module) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    lora = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.endswith(".lora_A") or name.endswith(".lora_B")
    )
    return {
        "total_parameters_with_adapter": total,
        "trainable_parameters": trainable,
        "lora_parameters": lora,
        "trainable_percentage": 100.0 * trainable / total if total else 0.0,
        "lora_modules": len(lora_module_names(model)),
    }


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    names = set(lora_parameter_names(model))
    if not names:
        raise RuntimeError("模型中没有 LoRA 参数")
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name in names
    }


def save_lora_adapter(
    model: nn.Module,
    output_dir: str | Path,
    *,
    base_model: str | None = None,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    wrappers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]
    if not wrappers:
        raise RuntimeError("模型中没有 LoRA 模块")
    if any(module.merged for _, module in wrappers):
        raise RuntimeError("保存 adapter 前必须先 unmerge LoRA")

    ranks = {module.rank for _, module in wrappers}
    alphas = {module.alpha for _, module in wrappers}
    dropout_values = {
        module.lora_dropout.p if isinstance(module.lora_dropout, nn.Dropout) else 0.0
        for _, module in wrappers
    }
    if len(ranks) != 1 or len(alphas) != 1 or len(dropout_values) != 1:
        raise RuntimeError("当前保存格式要求所有 LoRA 层使用相同超参数")

    config: dict[str, Any] = {
        "format_version": 1,
        "base_model": base_model,
        "rank": next(iter(ranks)),
        "alpha": next(iter(alphas)),
        "dropout": next(iter(dropout_values)),
        "target_suffixes": list(DEFAULT_TARGET_SUFFIXES),
        "modules": [name for name, _ in wrappers],
        "parameter_report": parameter_report(model),
    }
    if extra_config:
        config.update(extra_config)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(adapter_state_dict(model), output_path / ADAPTER_WEIGHTS_NAME)
    (output_path / ADAPTER_CONFIG_NAME).write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return config


def _load_weights(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # 兼容旧版 PyTorch。
        return torch.load(path, map_location="cpu")


@torch.no_grad()
def load_lora_adapter(
    model: nn.Module,
    adapter_dir: str | Path,
    *,
    inject_if_missing: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    adapter_path = Path(adapter_dir)
    config = json.loads(
        (adapter_path / ADAPTER_CONFIG_NAME).read_text(encoding="utf-8")
    )

    current_modules = lora_module_names(model)
    if not current_modules:
        if not inject_if_missing:
            raise RuntimeError("模型尚未注入 LoRA")
        current_modules = inject_lora(
            model,
            rank=int(config["rank"]),
            alpha=float(config["alpha"]),
            dropout=float(config.get("dropout", 0.0)),
            target_suffixes=config.get("target_suffixes", DEFAULT_TARGET_SUFFIXES),
        )

    expected_modules = list(config.get("modules", []))
    if strict and expected_modules and current_modules != expected_modules:
        raise RuntimeError(
            "LoRA 模块列表不匹配:\n"
            f"adapter={expected_modules}\nmodel={current_modules}"
        )

    state = _load_weights(adapter_path / ADAPTER_WEIGHTS_NAME)
    named_parameters = dict(model.named_parameters())
    expected_parameters = set(lora_parameter_names(model))
    state_parameters = set(state)
    if strict and expected_parameters != state_parameters:
        raise RuntimeError(
            "LoRA 参数键不匹配:\n"
            f"missing={sorted(expected_parameters - state_parameters)}\n"
            f"unexpected={sorted(state_parameters - expected_parameters)}"
        )

    for name, tensor in state.items():
        if name not in named_parameters:
            if strict:
                raise KeyError(f"模型中不存在 adapter 参数: {name}")
            continue
        parameter = named_parameters[name]
        if parameter.shape != tensor.shape:
            raise ValueError(
                f"参数形状不匹配 {name}: model={tuple(parameter.shape)}, "
                f"adapter={tuple(tensor.shape)}"
            )
        parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    return config


def merge_lora_weights(model: nn.Module, unload: bool = False) -> list[str]:
    wrappers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]
    for name, module in wrappers:
        module.merge()
        if unload:
            parent, child_name = _parent_and_child(model, name)
            setattr(parent, child_name, module.base_layer)
    return [name for name, _ in wrappers]


def unmerge_lora_weights(model: nn.Module) -> list[str]:
    wrappers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]
    for _, module in wrappers:
        module.unmerge()
    return [name for name, _ in wrappers]

