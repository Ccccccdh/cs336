"""自定义 LoRA 的单元测试；运行方式：python -m unittest tests.test_lora -v。"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from utils.lora import (
    LoRALinear,
    inject_lora,
    load_lora_adapter,
    lora_module_names,
    merge_lora_weights,
    parameter_report,
    save_lora_adapter,
    unmerge_lora_weights,
)


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(6, 10, bias=False)
        self.up_proj = nn.Linear(6, 10, bias=False)
        self.down_proj = nn.Linear(10, 6, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.gate_proj(inputs))
        hidden = hidden * self.up_proj(inputs)
        return self.down_proj(hidden)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyMLP(), TinyMLP()])
        self.untouched = nn.Linear(6, 6)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        return self.untouched(inputs)


class LoRATest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_initial_output_equals_base_linear(self) -> None:
        base = nn.Linear(7, 5, bias=True)
        inputs = torch.randn(3, 7)
        expected = base(inputs).detach().clone()
        layer = LoRALinear(base, rank=3, alpha=6, dropout=0.0)
        actual = layer(inputs)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_only_lora_parameters_receive_gradients(self) -> None:
        layer = LoRALinear(nn.Linear(7, 5), rank=2, alpha=4)
        loss = layer(torch.randn(4, 7)).square().mean()
        loss.backward()

        self.assertIsNone(layer.base_layer.weight.grad)
        self.assertIsNone(layer.base_layer.bias.grad)
        self.assertIsNotNone(layer.lora_A.grad)
        self.assertIsNotNone(layer.lora_B.grad)
        self.assertGreater(layer.lora_B.grad.abs().sum().item(), 0.0)

    def test_merge_and_unmerge_preserve_output(self) -> None:
        layer = LoRALinear(nn.Linear(7, 5), rank=3, alpha=6)
        with torch.no_grad():
            layer.lora_B.normal_(mean=0.0, std=0.1)
        inputs = torch.randn(4, 7)
        expected = layer(inputs).detach().clone()

        layer.merge()
        merged = layer(inputs)
        torch.testing.assert_close(merged, expected, rtol=1e-5, atol=1e-6)

        layer.unmerge()
        unmerged = layer(inputs)
        torch.testing.assert_close(unmerged, expected, rtol=1e-5, atol=1e-6)

    def test_inject_targets_only_ffn_projections(self) -> None:
        model = TinyModel()
        names = inject_lora(model, rank=4, alpha=8, dropout=0.05)
        self.assertEqual(len(names), 6)
        self.assertEqual(len(lora_module_names(model)), 6)
        self.assertIsInstance(model.untouched, nn.Linear)

        trainable_names = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        self.assertTrue(trainable_names)
        self.assertTrue(
            all(name.endswith(".lora_A") or name.endswith(".lora_B") for name in trainable_names)
        )
        report = parameter_report(model)
        expected = 2 * 4 * ((6 + 10) + (6 + 10) + (10 + 6))
        self.assertEqual(report["trainable_parameters"], expected)
        self.assertEqual(report["lora_parameters"], expected)

    def test_adapter_save_and_load(self) -> None:
        base = TinyModel()
        source = copy.deepcopy(base)
        target = copy.deepcopy(base)
        inject_lora(source, rank=2, alpha=4)
        with torch.no_grad():
            for module in source.modules():
                if isinstance(module, LoRALinear):
                    module.lora_A.normal_(mean=0.0, std=0.1)
                    module.lora_B.normal_(mean=0.0, std=0.1)

        inputs = torch.randn(2, 6)
        expected = source(inputs).detach().clone()
        with tempfile.TemporaryDirectory() as temp_dir:
            save_lora_adapter(source, temp_dir, base_model="tiny")
            config = load_lora_adapter(target, temp_dir)
            self.assertEqual(config["rank"], 2)
            self.assertTrue((Path(temp_dir) / "adapter_model.pt").exists())
        actual = target(inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_model_merge_and_unload(self) -> None:
        model = TinyModel()
        inject_lora(model, rank=2, alpha=4)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    module.lora_B.normal_(mean=0.0, std=0.1)
        inputs = torch.randn(2, 6)
        expected = model(inputs).detach().clone()

        merged_names = merge_lora_weights(model, unload=False)
        self.assertEqual(len(merged_names), 6)
        torch.testing.assert_close(model(inputs), expected, rtol=1e-5, atol=1e-6)

        unmerge_lora_weights(model)
        torch.testing.assert_close(model(inputs), expected, rtol=1e-5, atol=1e-6)

        merge_lora_weights(model, unload=True)
        self.assertFalse(lora_module_names(model))
        torch.testing.assert_close(model(inputs), expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
