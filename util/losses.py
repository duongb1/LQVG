import torch
import torch.nn.functional as F


def attention_alignment_loss(attn_probs: torch.Tensor, target_mask: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Compute a binary cross-entropy style alignment loss between attention and a visibility mask.

    Args:
        attn_probs: Tensor of shape [..., N] containing attention probabilities in [0, 1].
        target_mask: Tensor broadcastable to attn_probs containing ground-truth visibility in [0, 1].
        reduction: Reduction method passed to :func:`torch.nn.functional.binary_cross_entropy`.

    Returns:
        torch.Tensor: Loss value after applying the requested reduction.
    """
    if attn_probs.shape != target_mask.shape:
        target_mask = target_mask.expand_as(attn_probs)

    attn_probs = attn_probs.clamp(0.0, 1.0)
    target_mask = target_mask.clamp(0.0, 1.0)
    return F.binary_cross_entropy(attn_probs, target_mask, reduction=reduction)
