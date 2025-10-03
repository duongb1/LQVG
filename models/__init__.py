from .LQVG import build

from .lora_utils import (
    apply_conv_lora_1x1,
    apply_lora_linear,
    freeze_all_but_lora_and_heads,
    report_trainable,
)

# ----------------------------
# LoRA target patterns (regex)
# ----------------------------
# Text encoder (RoBERTa/BERT-style under model.text_encoder.*)
TEXT_LORA_PATTERNS = [
    r"^text_encoder\.encoder\.layer\.\d+\.attention\.self\.(query|key|value)$",
    r"^text_encoder\.encoder\.layer\.\d+\.attention\.output\.dense$",
    r"^text_encoder\.encoder\.layer\.\d+\.intermediate\.dense$",
    r"^text_encoder\.encoder\.layer\.\d+\.output\.dense$",
    r"^text_encoder\.pooler\.dense$",
]

# Fusion modules (these MHA call-sites expect a single tensor return)
FUSION_LORA_PATTERNS = [
    r"^fusion_module\.multihead_attn$",
    r"^fusion_module_text\.multihead_attn$",
]

# Decoder FFN & (optional) deformable cross-attn linears
# NOTE: DO NOT wrap 'decoder.self_attn' here; that call-site expects a (output, weights) tuple.
DECODER_LORA_PATTERNS = [
    r"^transformer\.decoder\.layers\.\d+\.linear1$",
    r"^transformer\.decoder\.layers\.\d+\.linear2$",
]

# MSDeformAttn (cross-attn) sub-linears — only if your MSDeformAttn exposes these attributes.
DECODER_DEFORM_PATTERNS = [
    r"^transformer\.decoder\.layers\.\d+\.cross_attn\.(sampling_offsets|attention_weights|value_proj|output_proj)$",
]

# Encoder FFN (safe to LoRA)
ENCODER_LORA_PATTERNS = [
    r"^transformer\.encoder\.layers\.\d+\.linear1$",
    r"^transformer\.encoder\.layers\.\d+\.linear2$",
]

# Encoder MSDeformAttn sub-linears
ENCODER_DEFORM_PATTERNS = [
    r"^transformer\.encoder\.layers\.\d+\.self_attn\.(sampling_offsets|attention_weights|value_proj|output_proj)$",
]

# Input projection convs (usually Sequential[i].0 is Conv2d 1x1)
INPUT_PROJ_PATTERNS = [
    r"^input_proj\.\d+\.0$",
]


def _resolve_lora_flags(args):
    """
    Derive LoRA enable flags from args.
    Presets:
      - light:    decoder FFN only
      - balanced: text + fusion + decoder
      - max:      text + fusion + decoder + encoder + input_proj
    """
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


def _log(title: str):
    print(f"[LoRA][init] {title}")


def build_model(args):
    model, criterion, postprocessors = build(args)

    if not getattr(args, "lora", False):
        return model, criterion, postprocessors

    rank = getattr(args, "lora_rank", 8)
    alpha = getattr(args, "lora_alpha", 32)
    dropout = getattr(args, "lora_dropout", 0.05)

    flags = _resolve_lora_flags(args)
    _log(f"preset resolved -> {flags}, r={rank}, alpha={alpha}, dropout={dropout}")

    # --- TEXT ---
    if flags.get("text"):
        _log("apply TEXT LoRA")
        apply_lora_linear(model, TEXT_LORA_PATTERNS, r=rank, alpha=alpha, dropout=dropout)

    # --- FUSION (MHA returns single tensor; our wrapper is compatible) ---
    if flags.get("fusion"):
        _log("apply FUSION LoRA")
        apply_lora_linear(model, FUSION_LORA_PATTERNS, r=rank, alpha=alpha, dropout=dropout)

    # --- DECODER (FFN + optional deformable) ---
    if flags.get("decoder"):
        _log("apply DECODER LoRA (FFN)")
        apply_lora_linear(model, DECODER_LORA_PATTERNS, r=rank, alpha=alpha, dropout=dropout)

        _log("apply DECODER MSDeformAttn sub-linears (if present)")
        apply_lora_linear(model, DECODER_DEFORM_PATTERNS, r=rank, alpha=alpha, dropout=dropout)

    # --- ENCODER (FFN + optional deformable) ---
    if flags.get("encoder"):
        _log("apply ENCODER LoRA (FFN)")
        apply_lora_linear(model, ENCODER_LORA_PATTERNS, r=rank, alpha=alpha, dropout=dropout)

        _log("apply ENCODER MSDeformAttn sub-linears (if present)")
        apply_lora_linear(model, ENCODER_DEFORM_PATTERNS, r=rank, alpha=alpha, dropout=dropout)

    # --- INPUT PROJ (Conv1x1) ---
    if flags.get("input_proj"):
        conv_rank = max(4, rank // 2)
        _log(f"apply INPUT_PROJ LoRA (conv1x1), r={conv_rank}")
        apply_conv_lora_1x1(model, INPUT_PROJ_PATTERNS, r=conv_rank, alpha=alpha)

    # Freeze everything except LoRA params + detection heads / queries / refs
    keep_keywords = ["class_embed", "bbox_embed", "query_embed", "reference_points"]
    freeze_all_but_lora_and_heads(model, keep_keywords=keep_keywords)
    report_trainable(model)

    return model, criterion, postprocessors
