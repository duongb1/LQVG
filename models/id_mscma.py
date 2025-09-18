
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

=======
# models/id_mscma.py
from typing import List, Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F

# internal deformable attention
try:
    from models.ops.modules import MSDeformAttn
except Exception:
    MSDeformAttn = None

from models.direction_parser import parse_direction_tokens, make_direction_fields_multi

# --------- Small blocks ---------
class FFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 4*d), nn.GELU(), nn.Dropout(0.1), nn.Linear(4*d, d)
        )
    def forward(self, x): return self.net(x)

class FiLMLite(nn.Module):
    def __init__(self, d_text: int, c_vis: int, init_alpha: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(d_text), nn.Linear(d_text, 2*c_vis), nn.GELU(), nn.Linear(2*c_vis, 2*c_vis))
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
    def forward(self, v: torch.Tensor, t_global: torch.Tensor) -> torch.Tensor:
        B,C,H,W = v.shape
        g,b = self.mlp(t_global).chunk(2, dim=-1)  # [B,C] x2
        g = g.unsqueeze(-1).unsqueeze(-1); b = b.unsqueeze(-1).unsqueeze(-1)
        return v * (1.0 + self.alpha * g) + self.alpha * b

# --------- Deformable adapter for LVI ---------
class MSDeformAttnAdapter(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=4, n_points=4):
        super().__init__()
        if MSDeformAttn is None:
            raise ImportError("MSDeformAttn not found. Import from internal path models/ops/modules/*.py")
        self.attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        q: torch.Tensor,
        feats: List[torch.Tensor],
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        reference_points: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(q)
        v = torch.cat([f.flatten(2).transpose(1, 2) for f in feats], dim=1)  # [B,sum(HiWi),C]
        out, _, _ = self.attn(q, reference_points, v, spatial_shapes, level_start_index, None)
        return self.out_proj(out)

# --------- One loop: LVI -> VLI ---------
class BiDirCMLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=4, n_points=4):
        super().__init__()
        self.lvi = MSDeformAttnAdapter(d_model, n_heads, n_levels, n_points)
        self.lvi_norm1 = nn.LayerNorm(d_model); self.lvi_ffn = FFN(d_model); self.lvi_norm2 = nn.LayerNorm(d_model)
        self.vli_attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.vli_norm1 = nn.LayerNorm(d_model); self.vli_ffn = FFN(d_model); self.vli_norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def _apply_dir_offset(
        ref_pts: torch.Tensor,
        dir_vecs: Optional[torch.Tensor],
        mag: float,
    ) -> torch.Tensor:
        if dir_vecs is None or mag <= 0:
            return ref_pts
        if dir_vecs.ndim == 1:
            dir_vecs = dir_vecs.unsqueeze(0)
        if dir_vecs.shape[0] != ref_pts.shape[0]:
            raise ValueError("Direction vector batch size must match reference points batch size")
        if not torch.any(dir_vecs != 0):
            return ref_pts
        off = ref_pts.clone()
        dx = dir_vecs[:, 0].view(-1, 1, 1, 1)
        dy = dir_vecs[:, 1].view(-1, 1, 1, 1)
        off[..., 0] = torch.clamp(off[..., 0] + mag * dx, 0.0, 1.0)
        off[..., 1] = torch.clamp(off[..., 1] + mag * dy, 0.0, 1.0)
        return off

    def forward(self, txt_tokens: torch.Tensor, vis_tokens: torch.Tensor, feats: List[torch.Tensor],
                spatial_shapes: torch.Tensor, level_start_index: torch.Tensor,
                reference_points: torch.Tensor,
                dir_field_seq: Optional[torch.Tensor], dir_vecs: Optional[torch.Tensor],
                lambda_dir_lvi: float, lambda_dir_vli: float,
                txt_pos: Optional[torch.Tensor]=None, vis_pos: Optional[torch.Tensor]=None):
        # ----- LVI (text <- vision, deformable) -----
        t = txt_tokens + (txt_pos if txt_pos is not None else 0)
        ref = self._apply_dir_offset(reference_points, dir_vecs, mag=0.08 * lambda_dir_lvi)
        t_attn = self.lvi(t, feats, spatial_shapes, level_start_index, ref)  # [B,L,C]
        t2 = self.lvi_norm1(txt_tokens + t_attn)
        t2 = self.lvi_norm2(t2 + self.lvi_ffn(t2))

        # ----- VLI (vision <- text, dense) -----
        q = vis_tokens + (vis_pos if vis_pos is not None else 0)
        k = t2 + (txt_pos if txt_pos is not None else 0)
        out, attn_w = self.vli_attn(q, k, k, need_weights=True)  # attn_w: [B,N,L]
        if dir_field_seq is not None and lambda_dir_vli > 0:
            bias = dir_field_seq.transpose(1, 2).to(out.dtype)  # [B, N, 1]
            out = out * (1.0 + lambda_dir_vli * (bias - 0.5))
        v2 = self.vli_norm1(vis_tokens + out)
        v2 = self.vli_norm2(v2 + self.vli_ffn(v2))
        return t2, v2, attn_w

# --------- Iterative ID-MSCMA (N=2) ---------
class ID_MSCMA(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=4, n_points=4, n_layers=2, weight_sharing=True):
        super().__init__()
        self.n_layers = n_layers; self.weight_sharing = weight_sharing
        if weight_sharing:
            self.layer = BiDirCMLayer(d_model, n_heads, n_levels, n_points)
        else:
            self.layers = nn.ModuleList([BiDirCMLayer(d_model, n_heads, n_levels, n_points) for _ in range(n_layers)])

    def forward(self, txt_tokens: torch.Tensor, feats: List[torch.Tensor],
                spatial_shapes: torch.Tensor, level_start_index: torch.Tensor,
                reference_points: torch.Tensor, vis_tokens: Optional[torch.Tensor]=None,
                dir_tokens: Optional[List[List[str]]]=None, anneal=(0.3, 0.1)):
        if vis_tokens is None:
            vis_tokens = torch.cat([f.flatten(2).transpose(1, 2) for f in feats], dim=1)
        dir_vecs = None
        dir_field_seq = None
        if dir_tokens is not None:
            batch = txt_tokens.shape[0]
            if len(dir_tokens) != batch:
                raise ValueError(
                    f"Expected {batch} direction token lists, but received {len(dir_tokens)}"
                )
            device = feats[0].device
            dtype = feats[0].dtype
            num_vis_tokens = int(spatial_shapes.prod(1).sum().item())
            neutral_field = torch.full((1, 1, num_vis_tokens), 0.5, dtype=dtype, device=device)
            parsed_vecs: List[Tuple[float, float]] = []
            field_seqs: List[torch.Tensor] = []
            has_any = False
            for tokens in dir_tokens:
                vec = parse_direction_tokens(tokens) if tokens is not None else None
                if vec is not None:
                    has_any = True
                    parsed_vecs.append((float(vec[0]), float(vec[1])))
                    fields = make_direction_fields_multi(feats, vec, device=device)
                    field_seqs.append(torch.cat(fields, dim=-1).to(dtype))
                else:
                    parsed_vecs.append((0.0, 0.0))
                    field_seqs.append(neutral_field.clone())
            if has_any:
                dir_vecs = torch.tensor(parsed_vecs, dtype=reference_points.dtype, device=device)
                dir_field_seq = torch.cat(field_seqs, dim=0)
            else:
                dir_vecs = None
                dir_field_seq = None
        t, v = txt_tokens, vis_tokens
        last_attn = None
        for i in range(self.n_layers):
            layer = self.layer if self.weight_sharing else self.layers[i]
            lam = anneal[0] if i == 0 else anneal[1]
            t, v, last_attn = layer(
                t, v, feats, spatial_shapes, level_start_index, reference_points,
                dir_field_seq=dir_field_seq, dir_vecs=dir_vecs,
                lambda_dir_lvi=lam, lambda_dir_vli=lam
            )
        return t, v, last_attn

# --------- AAL (attention alignment loss) ---------
def attention_alignment_loss(attn_weights: torch.Tensor, gt_mask: torch.Tensor, reduction="mean"):
    """Compute BCE on attention probabilities against ground-truth visibility masks."""
    if gt_mask.dim() == 3 and gt_mask.size(1) == 1:
        gt_mask = gt_mask.squeeze(1)
    elif gt_mask.dim() != 2:
        raise ValueError("gt_mask must have shape [B, N] or [B, 1, N]")
    probs = attn_weights.mean(dim=2).clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(probs, gt_mask.to(probs.dtype), reduction=reduction)
