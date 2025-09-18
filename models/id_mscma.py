"""Direction-aware prior modelling for LQVG.

This module implements a lightweight approximation of the ID-MSCMA
component mentioned in the project description.  The block parses natural
language direction cues, converts them into attention priors and injects
them into the query embeddings that drive the transformer decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn

# Mapping from lowercase keywords to coarse direction vectors.  The
# directions are expressed in an (x, y, z) basis where the axes correspond to
# horizontal, vertical and depth cues respectively.
_DIRECTION_VECTORS = {
    "left": torch.tensor([-1.0, 0.0, 0.0]),
    "right": torch.tensor([1.0, 0.0, 0.0]),
    "up": torch.tensor([0.0, 1.0, 0.0]),
    "above": torch.tensor([0.0, 1.0, 0.0]),
    "top": torch.tensor([0.0, 1.0, 0.0]),
    "down": torch.tensor([0.0, -1.0, 0.0]),
    "below": torch.tensor([0.0, -1.0, 0.0]),
    "bottom": torch.tensor([0.0, -1.0, 0.0]),
    "front": torch.tensor([0.0, 0.0, 1.0]),
    "forward": torch.tensor([0.0, 0.0, 1.0]),
    "towards": torch.tensor([0.0, 0.0, 1.0]),
    "ahead": torch.tensor([0.0, 0.0, 1.0]),
    "back": torch.tensor([0.0, 0.0, -1.0]),
    "backward": torch.tensor([0.0, 0.0, -1.0]),
    "behind": torch.tensor([0.0, 0.0, -1.0]),
    "away": torch.tensor([0.0, 0.0, -1.0]),
}


@dataclass
class DirectionParseResult:
    """Simple container for the output of :func:`parse_direction_tokens`."""

    vectors: Tensor
    mask: Tensor


def parse_direction_tokens(
    batch_tokens: Sequence[Sequence[str]],
    device: torch.device | None = None,
) -> DirectionParseResult:
    """Convert direction keywords into coarse attention priors.

    Args:
        batch_tokens: A sequence of token sequences.  Each inner sequence
            contains lowercase tokens derived from a caption belonging to the
            corresponding batch element.
        device: Optional target device for the resulting tensors.

    Returns:
        A :class:`DirectionParseResult` containing a tensor of shape
        ``[B, 3]`` with aggregated direction vectors and a mask ``[B]`` whose
        entries are ``1`` when the associated caption produced at least one
        direction cue and ``0`` otherwise.
    """

    if not isinstance(batch_tokens, Sequence):
        raise TypeError("batch_tokens is expected to be a sequence of token lists")

    direction_vectors: List[Tensor] = []
    mask: List[float] = []

    for tokens in batch_tokens:
        if tokens is None:
            tokens = []
        if not isinstance(tokens, Iterable):
            raise TypeError("Each element in batch_tokens must be an iterable of strings")

        aggregate = torch.zeros(3)
        found_any = False

        for token in tokens:
            if not isinstance(token, str):
                continue
            cleaned = token.lower().strip(".,!?\"'()[]{}")
            if not cleaned:
                continue
            if cleaned in _DIRECTION_VECTORS:
                aggregate = aggregate + _DIRECTION_VECTORS[cleaned]
                found_any = True

        direction_vectors.append(aggregate)
        mask.append(1.0 if found_any else 0.0)

    vectors = torch.stack(direction_vectors, dim=0)
    mask_tensor = torch.tensor(mask, dtype=torch.float32)

    if device is not None:
        vectors = vectors.to(device)
        mask_tensor = mask_tensor.to(device)

    return DirectionParseResult(vectors=vectors, mask=mask_tensor)


class BiDirCMLayer(nn.Module):
    """Bidirectional cross-modal layer used inside ID-MSCMA.

    The layer consumes the sentence-level features alongside direction priors
    and produces feature refinements together with attention biases.  The core
    of the layer is :meth:`_apply_dir_offset` which turns the encoded direction
    signal into additive offsets that are later injected into the transformer
    queries.
    """

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.dir_offset = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.head_proj = nn.Linear(hidden_dim, num_heads)

    def forward(
        self,
        sentence_features: Tensor,
        encoded_direction: Tensor,
        direction_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if sentence_features.dim() != 2:
            raise ValueError(
                "sentence_features must be of shape [B, C], got "
                f"{tuple(sentence_features.shape)}"
            )

        dir_offset = self._apply_dir_offset(encoded_direction)
        dir_offset = dir_offset * direction_mask.unsqueeze(-1)
        refined_features = sentence_features + dir_offset
        attn_bias = self.head_proj(dir_offset)
        return refined_features, dir_offset, attn_bias

    def _apply_dir_offset(self, encoded_direction: Tensor) -> Tensor:
        return self.dir_offset(encoded_direction)


class ID_MSCMA(nn.Module):
    """Inter-frame Directional Multi-Scale Cross-Modal Attention."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.direction_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.cm_layer = BiDirCMLayer(hidden_dim=hidden_dim, num_heads=num_heads)

    def forward(
        self,
        sentence_features: Tensor,
        direction_tokens: Sequence[Sequence[str]],
    ) -> dict[str, Tensor]:
        if sentence_features.dim() != 2:
            raise ValueError(
                "sentence_features must be 2-D with shape [B, C], got "
                f"{tuple(sentence_features.shape)}"
            )

        device = sentence_features.device
        parse_result = parse_direction_tokens(direction_tokens, device=device)

        encoded_direction = self.direction_encoder(parse_result.vectors)
        encoded_direction = encoded_direction * parse_result.mask.unsqueeze(-1)

        refined_features, dir_offset, attn_bias = self.cm_layer(
            sentence_features, encoded_direction, parse_result.mask
        )

        return {
            "refined_features": refined_features,
            "dir_offset": dir_offset,
            "attn_bias": attn_bias,
            "mask": parse_result.mask,
        }

