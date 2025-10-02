from .LQVG import build

from .lora_utils import (
    apply_conv_lora_1x1,
    apply_lora_linear,
    freeze_all_but_lora_and_heads,
    report_trainable,
)

TEXT_LORA_PATTERNS = [
    r"^text_encoder\.encoder\.layer\.\d+\.attention\.self\.(query|key|value)$",
    r"^text_encoder\.encoder\.layer\.\d+\.attention\.output\.dense$",
    r"^text_encoder\.encoder\.layer\.\d+\.intermediate\.dense$",
    r"^text_encoder\.encoder\.layer\.\d+\.output\.dense$",
    r"^text_encoder\.pooler\.dense$",
]

FUSION_LORA_PATTERNS = [
    r"^fusion_module\.multihead_attn\.out_proj$",
    r"^fusion_module_text\.multihead_attn\.out_proj$",
]

DECODER_LORA_PATTERNS = [
    r"^transformer\.decoder\.layers\.\d+\.self_attn\.out_proj$",
    r"^transformer\.decoder\.layers\.\d+\.linear1$",
    r"^transformer\.decoder\.layers\.\d+\.linear2$",
]

DECODER_DEFORM_PATTERNS = [
    r"^transformer\.decoder\.layers\.\d+\.cross_attn\.(sampling_offsets|attention_weights|value_proj|output_proj)$",
]

ENCODER_LORA_PATTERNS = [
    r"^transformer\.encoder\.layers\.\d+\.linear1$",
    r"^transformer\.encoder\.layers\.\d+\.linear2$",
]

ENCODER_DEFORM_PATTERNS = [
    r"^transformer\.encoder\.layers\.\d+\.self_attn\.(sampling_offsets|attention_weights|value_proj|output_proj)$",
]

INPUT_PROJ_PATTERNS = [
    r"^input_proj\.\d+\.0$",
]


def _resolve_lora_flags(args):
    preset = getattr(args, "lora_preset", "balanced")
    flags = {
        "text": getattr(args, "lora_text", False),
        "fusion": getattr(args, "lora_fusion", False),
        "decoder": getattr(args, "lora_decoder", False),
        "encoder": getattr(args, "lora_encoder", False),
        "input_proj": getattr(args, "lora_input_proj", False),
    }
    if not any(flags.values()):
        if preset == "light":
            flags["decoder"] = True
        elif preset == "balanced":
            flags.update({"text": True, "fusion": True, "decoder": True})
        elif preset == "max":
            for key in flags:
                flags[key] = True
    return flags


def build_model(args):
    model, criterion, postprocessors = build(args)

    if not getattr(args, "lora", False):
        return model, criterion, postprocessors

    rank = getattr(args, "lora_rank", 8)
    alpha = getattr(args, "lora_alpha", 32)
    dropout = getattr(args, "lora_dropout", 0.05)

    flags = _resolve_lora_flags(args)

    if flags.get("text"):
        apply_lora_linear(model, TEXT_LORA_PATTERNS, r=rank, alpha=alpha, dropout=dropout)

    if flags.get("fusion"):
        apply_lora_linear(model, FUSION_LORA_PATTERNS, r=rank, alpha=alpha, dropout=dropout)

    if flags.get("decoder"):
        apply_lora_linear(model, DECODER_LORA_PATTERNS, r=rank, alpha=alpha, dropout=dropout)
        apply_lora_linear(model, DECODER_DEFORM_PATTERNS, r=rank, alpha=alpha, dropout=dropout)

    if flags.get("encoder"):
        apply_lora_linear(model, ENCODER_LORA_PATTERNS, r=rank, alpha=alpha, dropout=dropout)
        apply_lora_linear(model, ENCODER_DEFORM_PATTERNS, r=rank, alpha=alpha, dropout=dropout)

    if flags.get("input_proj"):
        conv_rank = max(4, rank // 2)
        apply_conv_lora_1x1(model, INPUT_PROJ_PATTERNS, r=conv_rank, alpha=alpha)

    keep_keywords = ["class_embed", "bbox_embed", "query_embed", "reference_points"]
    freeze_all_but_lora_and_heads(model, keep_keywords=keep_keywords)
    report_trainable(model)

    return model, criterion, postprocessors
