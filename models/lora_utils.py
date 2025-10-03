import re
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:  # pragma: no cover - optional dependency
    from peft import LoraConfig, TaskType, get_peft_model

    _HAS_PEFT = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_PEFT = False


class LoRALinear(nn.Module):
    def __init__(
        self, linear: nn.Linear, r: int = 8, alpha: int = 32, dropout: float = 0.0
    ):
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
        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
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
        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)
        self.scaling = alpha / max(1, r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.B(self.A(x)) * self.scaling


class LoRAMultiheadAttention(nn.Module):
    def __init__(
        self,
        attn: nn.MultiheadAttention,
        r: int = 8,
        alpha: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        if attn.batch_first:
            raise ValueError(
                "LoRAMultiheadAttention does not currently support batch_first=True"
            )
        if attn.bias_k is not None or attn.bias_v is not None:
            raise ValueError("LoRAMultiheadAttention does not support bias_k or bias_v")

        self.embed_dim = attn.embed_dim
        self.num_heads = attn.num_heads
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout_p = attn.dropout

        in_proj_bias = attn.in_proj_bias is not None
        weight = attn.in_proj_weight.detach()
        bias = attn.in_proj_bias.detach() if in_proj_bias else None

        q_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=in_proj_bias)
        k_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=in_proj_bias)
        v_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=in_proj_bias)

        q_linear.weight.data.copy_(weight[: self.embed_dim])
        k_linear.weight.data.copy_(weight[self.embed_dim : 2 * self.embed_dim])
        v_linear.weight.data.copy_(weight[2 * self.embed_dim :])
        if in_proj_bias:
            q_linear.bias.data.copy_(bias[: self.embed_dim])
            k_linear.bias.data.copy_(bias[self.embed_dim : 2 * self.embed_dim])
            v_linear.bias.data.copy_(bias[2 * self.embed_dim :])

        out_linear = nn.Linear(
            self.embed_dim, self.embed_dim, bias=attn.out_proj.bias is not None
        )
        out_linear.weight.data.copy_(attn.out_proj.weight.detach())
        if attn.out_proj.bias is not None:
            out_linear.bias.data.copy_(attn.out_proj.bias.detach())

        self.q_proj = LoRALinear(q_linear, r=r, alpha=alpha, dropout=dropout)
        self.k_proj = LoRALinear(k_linear, r=r, alpha=alpha, dropout=dropout)
        self.v_proj = LoRALinear(v_linear, r=r, alpha=alpha, dropout=dropout)
        self.out_proj = LoRALinear(out_linear, r=r, alpha=alpha, dropout=dropout)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        # x: (L, N, C) -> (N, heads, L, head_dim)
        L, N, _ = x.shape
        x = x.permute(1, 0, 2).contiguous().view(N, L, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, heads, L, head_dim) -> (L, N, C)
        N, _, L, _ = x.shape
        x = x.transpose(1, 2).contiguous().view(N, L, self.embed_dim)
        return x.permute(1, 0, 2)

    def forward(
        self,
        query: torch.Tensor = None,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,                  # mặc định True như torch MHA
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,                    # không dùng, để tương thích
        # --- tương thích legacy wrappers dùng tham số x ---
        x: torch.Tensor = None,
        **kwargs,
    ):
        # map legacy signature (x, key, value, ...) -> (query, key, value, ...)
        if query is None and x is not None:
            query = x
        if key is None:
            key = query
        if value is None:
            value = key
    
        # QKV chiếu LoRA
        q = self.q_proj(query)   # (Lq, N, C)
        k = self.k_proj(key)     # (Lk, N, C)
        v = self.v_proj(value)   # (Lv, N, C)
    
        # Kích thước
        Lq, Nq, _ = q.shape
        Lk, Nk, _ = k.shape
        Lv, Nv, _ = v.shape
        assert Nq == Nk == Nv, "Batch size (N) của q/k/v phải giống nhau"
    
        # reshape -> (N, heads, L*, head_dim)
        scaling = float(self.head_dim) ** -0.5
        q = (q.permute(1, 0, 2).contiguous().view(Nq, Lq, self.num_heads, self.head_dim).transpose(1, 2)) * scaling
        k =  k.permute(1, 0, 2).contiguous().view(Nk, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v =  v.permute(1, 0, 2).contiguous().view(Nv, Lv, self.num_heads, self.head_dim).transpose(1, 2)
    
        # logits: (N, heads, Lq, Lk)
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
    
        # ---- attn_mask ----
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # bool mask: True = masked
                if attn_mask.dim() == 2 and attn_mask.shape == (Lq, Lk):
                    mask = attn_mask.unsqueeze(0).unsqueeze(0)                       # (1,1,Lq,Lk)
                    fill_value = torch.finfo(attn_logits.dtype).min
                    attn_logits = attn_logits.masked_fill(mask, fill_value)
                elif attn_mask.dim() == 3:
                    if attn_mask.shape[0] == Nq * self.num_heads:                    # (N*H,Lq,Lk)
                        mask = attn_mask.view(Nq, self.num_heads, Lq, Lk)
                        fill_value = torch.finfo(attn_logits.dtype).min
                        attn_logits = attn_logits.masked_fill(mask, fill_value)
                    elif attn_mask.shape[0] == Nq:                                   # (N,Lq,Lk)
                        mask = attn_mask.unsqueeze(1)                                # (N,1,Lq,Lk)
                        fill_value = torch.finfo(attn_logits.dtype).min
                        attn_logits = attn_logits.masked_fill(mask, fill_value)
                    else:
                        raise ValueError("attn_mask bool 3D phải có batch N hoặc N*heads")
                else:
                    raise ValueError("attn_mask(bool) phải là (Lq,Lk) hoặc (N,Lq,Lk) hoặc (N*heads,Lq,Lk)")
            else:
                # float mask: cộng trực tiếp
                if attn_mask.dim() == 2 and attn_mask.shape == (Lq, Lk):
                    attn_logits = attn_logits + attn_mask.unsqueeze(0).unsqueeze(0)  # (1,1,Lq,Lk)
                elif attn_mask.dim() == 3:
                    if attn_mask.shape[0] == Nq * self.num_heads:                    # (N*H,Lq,Lk)
                        m = attn_mask.view(Nq, self.num_heads, Lq, Lk)
                        attn_logits = attn_logits + m
                    elif attn_mask.shape[0] == Nq:                                   # (N,Lq,Lk)
                        attn_logits = attn_logits + attn_mask.unsqueeze(1)          # (N,1,Lq,Lk)
                    else:
                        raise ValueError("attn_mask(float) 3D phải có batch N hoặc N*heads")
                else:
                    raise ValueError("attn_mask(float) phải là (Lq,Lk) hoặc (N,Lq,Lk) hoặc (N*heads,Lq,Lk)")
    
        # ---- key_padding_mask (N, Lk): True = pad -> -inf ----
        pad_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.shape != (Nq, Lk):
                raise ValueError(f"key_padding_mask phải có shape (N, Lk) = ({Nq},{Lk}), nhận {tuple(key_padding_mask.shape)}")
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (N,1,1,Lk)
            fill_value = torch.finfo(attn_logits.dtype).min
            attn_logits = attn_logits.masked_fill(pad_mask, fill_value)
    
        # softmax ổn định + dropout
        attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True).values
        attn_weights = torch.softmax(attn_logits, dim=-1)          # (N, heads, Lq, Lk)
    
        # re-normalize nếu có pad_mask (đảm bảo tổng = 1)
        if pad_mask is not None:
            attn_weights = attn_weights.masked_fill(pad_mask, 0.0)
            denom = attn_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(attn_weights.dtype).tiny)
            attn_weights = attn_weights / denom
    
        attn_weights = torch.dropout(attn_weights, self.dropout_p, self.training)
    
        # output: (N, heads, Lq, head_dim) -> (Lq, N, C)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(Nq, Lq, self.embed_dim).permute(1, 0, 2)
        attn_output = self.out_proj(attn_output)
    
        # Trả về giống nn.MultiheadAttention
        if not need_weights:
            return attn_output, None
    
        if average_attn_weights:
            # (N, heads, Lq, Lk) -> (N, Lq, Lk) -> (Lq, N, Lk)
            attn_weights_out = attn_weights.mean(dim=1).permute(1, 0, 2).contiguous()
        else:
            # giữ nguyên (N, heads, Lq, Lk)
            attn_weights_out = attn_weights
    
        return attn_output, attn_weights_out



def _match_any(name: str, regex_list: Sequence[str]) -> bool:
    return any(re.match(pattern, name) for pattern in regex_list)


def _get_parent_module(model: nn.Module, name: str):
    if not name:
        return None, ""
    parts = name.split(".")
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
    parts = name.split(".")
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


def apply_lora_linear(
    model: nn.Module,
    regex_list: Sequence[str],
    r: int = 8,
    alpha: int = 32,
    dropout: float = 0.05,
) -> None:
    if not regex_list:
        return
    replaced_set = set()
    named_modules = _collect_named_modules(model)

    if _HAS_PEFT:
        target_modules: List[str] = []
        for name, module in named_modules:
            # CHỈ Linear cho PEFT (loại bỏ MultiheadAttention ở đây)
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
        if not _match_any(name, regex_list):
            continue

        current_module = _get_module(model, name)

        if isinstance(current_module, LoRALinear):
            continue

        if isinstance(current_module, nn.MultiheadAttention):
            lora_module = LoRAMultiheadAttention(
                current_module, r=r, alpha=alpha, dropout=dropout
            )
            _set_module(model, name, lora_module)
            replaced_set.add(name)
            continue

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


def apply_conv_lora_1x1(
    model: nn.Module, regex_list: Sequence[str], r: int = 4, alpha: int = 32
) -> None:
    if not regex_list:
        return
    replaced_set = set()
    for name, module in _collect_named_modules(model):
        if (
            isinstance(module, nn.Conv2d)
            and module.kernel_size == (1, 1)
            and _match_any(name, regex_list)
        ):
            lora_module = ConvLoRA1x1(module, r=r, alpha=alpha)
            _set_module(model, name, lora_module)
            replaced_set.add(name)
    replaced = sorted(replaced_set)
    print(f"[LoRA] Conv1x1 wrapped: {len(replaced)}")
    for entry in replaced[:10]:
        print(f"  - {entry}")
    if len(replaced) > 10:
        print("  ...")


def freeze_all_but_lora_and_heads(
    model: nn.Module, keep_keywords: Iterable[str] = ()
):  # noqa: D401
    for param in model.parameters():
        param.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, LoRALinear):
            for name, param in module.named_parameters():
                if name.startswith("A") or name.startswith("B"):
                    param.requires_grad_(True)
        elif isinstance(module, ConvLoRA1x1):
            for name, param in module.named_parameters():
                if name.startswith("A") or name.startswith("B"):
                    param.requires_grad_(True)

    keep_keywords = tuple(keep_keywords or ())
    if keep_keywords:
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in keep_keywords):
                param.requires_grad_(True)


def report_trainable(model: nn.Module) -> None:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    pct = 100.0 * trainable / max(1, total)
    print(f"[LoRA] Trainable params: {trainable}/{total} ({pct:.2f}%)")


__all__ = [
    "LoRALinear",
    "ConvLoRA1x1",
    "apply_lora_linear",
    "apply_conv_lora_1x1",
    "freeze_all_but_lora_and_heads",
    "report_trainable",
]
