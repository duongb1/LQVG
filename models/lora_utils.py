import re
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn

try:  # pragma: no cover - optional dependency
    from peft import LoraConfig, TaskType, get_peft_model
    _HAS_PEFT = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_PEFT = False


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear module")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.weight = linear.weight
        self.bias = linear.bias
        if self.weight is not None:
            self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.A = nn.Linear(self.in_features, r, bias=False)
        self.B = nn.Linear(r, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)
        self.scaling = self.alpha / max(1, self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = nn.functional.linear(x, self.weight, self.bias)
        lora = self.B(self.A(self.dropout(x))) * self.scaling
        return base + lora


class ConvLoRA1x1(nn.Module):
    def __init__(self, conv1x1: nn.Conv2d, r: int = 4, alpha: int = 32):
        super().__init__()
        if not isinstance(conv1x1, nn.Conv2d) or conv1x1.kernel_size != (1, 1):
            raise TypeError("ConvLoRA1x1 expects a 1x1 nn.Conv2d module")
        self.conv = conv1x1
        self.conv.requires_grad_(False)
        in_channels = conv1x1.in_channels
        out_channels = conv1x1.out_channels
        self.A = nn.Conv2d(in_channels, r, kernel_size=1, bias=False)
        self.B = nn.Conv2d(r, out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)
        self.scaling = alpha / max(1, r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.B(self.A(x)) * self.scaling


def _match_any(name: str, regex_list: Sequence[str]) -> bool:
    return any(re.match(pattern, name) for pattern in regex_list)


def _get_parent_module(model: nn.Module, name: str):
    if not name:
        return None, ''
    parts = name.split('.')
    module = model
    for part in parts[:-1]:
        if part.isdigit() and isinstance(module, (nn.Sequential, nn.ModuleList, list)):
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module, parts[-1]


def _get_module(model: nn.Module, name: str) -> nn.Module:
    if not name:
        return model
    parts = name.split('.')
    module = model
    for part in parts:
        if part.isdigit() and isinstance(module, (nn.Sequential, nn.ModuleList, list)):
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _set_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parent, attr = _get_parent_module(model, name)
    if parent is None:
        raise ValueError(f"Cannot set module for root name '{name}'")
    if attr.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList, list)):
        parent[int(attr)] = new_module
    else:
        setattr(parent, attr, new_module)


def _collect_named_modules(model: nn.Module) -> List[tuple]:
    return list(model.named_modules())


def apply_lora_linear(model: nn.Module, regex_list: Sequence[str], r: int = 8, alpha: int = 32, dropout: float = 0.05) -> None:
    if not regex_list:
        return
    replaced_set = set()
    named_modules = _collect_named_modules(model)

    if _HAS_PEFT:
        target_modules: List[str] = []
        for name, module in named_modules:
            if isinstance(module, nn.Linear) and _match_any(name, regex_list):
                target_modules.append(name)
        if target_modules:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=target_modules,
            )
            get_peft_model(model, peft_config)
            replaced_set.update(target_modules)

    # Manual fallback or additional wrapping for fine-grained control
    for name, module in named_modules:
        if not isinstance(module, nn.Linear) or not _match_any(name, regex_list):
            continue
        current_module = _get_module(model, name)
        if not isinstance(current_module, nn.Linear):
            continue
        lora_module = LoRALinear(current_module, r=r, alpha=alpha, dropout=dropout)
        _set_module(model, name, lora_module)
        replaced_set.add(name)

    replaced = sorted(replaced_set)
    print(f"[LoRA] Linear wrapped: {len(replaced)}")
    for entry in replaced[:10]:
        print(f"  - {entry}")
    if len(replaced) > 10:
        print("  ...")


def apply_conv_lora_1x1(model: nn.Module, regex_list: Sequence[str], r: int = 4, alpha: int = 32) -> None:
    if not regex_list:
        return
    replaced_set = set()
    for name, module in _collect_named_modules(model):
        if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1) and _match_any(name, regex_list):
            lora_module = ConvLoRA1x1(module, r=r, alpha=alpha)
            _set_module(model, name, lora_module)
            replaced_set.add(name)
    replaced = sorted(replaced_set)
    print(f"[LoRA] Conv1x1 wrapped: {len(replaced)}")
    for entry in replaced[:10]:
        print(f"  - {entry}")
    if len(replaced) > 10:
        print("  ...")


def freeze_all_but_lora_and_heads(model: nn.Module, keep_keywords: Iterable[str] = ()):  # noqa: D401
    for param in model.parameters():
        param.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, LoRALinear):
            for name, param in module.named_parameters():
                if name.startswith('A') or name.startswith('B'):
                    param.requires_grad_(True)
        elif isinstance(module, ConvLoRA1x1):
            for name, param in module.named_parameters():
                if name.startswith('A') or name.startswith('B'):
                    param.requires_grad_(True)

    keep_keywords = tuple(keep_keywords or ())
    if keep_keywords:
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in keep_keywords):
                param.requires_grad_(True)


def report_trainable(model: nn.Module) -> None:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    pct = 100.0 * trainable / max(1, total)
    print(f"[LoRA] Trainable params: {trainable}/{total} ({pct:.2f}%)")


__all__ = [
    'LoRALinear',
    'ConvLoRA1x1',
    'apply_lora_linear',
    'apply_conv_lora_1x1',
    'freeze_all_but_lora_and_heads',
    'report_trainable',
]
