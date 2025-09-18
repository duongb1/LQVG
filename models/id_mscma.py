import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.ops.modules import MSDeformAttn
except Exception:  # pragma: no cover - fallback if ops unavailable at import time
    MSDeformAttn = None


class FFN(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMLite(nn.Module):
    def __init__(self, d_text: int, c_vis: int, init_alpha: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_text),
            nn.Linear(d_text, 2 * c_vis),
            nn.GELU(),
            nn.Linear(2 * c_vis, 2 * c_vis),
        )
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.mlp(t).chunk(2, dim=-1)
        gamma = gamma[..., None, None]
        beta = beta[..., None, None]
        return v * (1 + self.alpha * gamma) + self.alpha * beta


class MSDeformAttnAdapter(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_levels: int = 4, n_points: int = 4) -> None:
        super().__init__()
        if MSDeformAttn is None:
            raise ImportError("MSDeformAttn not found.")
        self.attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        q: torch.Tensor,
        feats: list[torch.Tensor],
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        reference_points: torch.Tensor,
    ) -> torch.Tensor:
        q_proj = self.q_proj(q)
        value = torch.cat([feat.flatten(2).transpose(1, 2) for feat in feats], dim=1)
        out = self.attn(q_proj, reference_points, value, spatial_shapes, level_start_index, None)
        if isinstance(out, tuple):
            out = out[0]
        return self.out_proj(out)


class BiDirCMLayer(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_levels: int = 4, n_points: int = 4) -> None:
        super().__init__()
        self.lvi = MSDeformAttnAdapter(d_model, n_heads, n_levels, n_points)
        self.lvi_norm1 = nn.LayerNorm(d_model)
        self.lvi_ffn = FFN(d_model)
        self.lvi_norm2 = nn.LayerNorm(d_model)

        self.vli_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.vli_norm1 = nn.LayerNorm(d_model)
        self.vli_ffn = FFN(d_model)
        self.vli_norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        txt_tokens: torch.Tensor,
        vis_tokens: torch.Tensor,
        feats: list[torch.Tensor],
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        reference_points: torch.Tensor,
        txt_pos: torch.Tensor | None = None,
        vis_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        txt_query = txt_tokens + (txt_pos if txt_pos is not None else 0)
        txt_attn = self.lvi(txt_query, feats, spatial_shapes, level_start_index, reference_points)
        txt_res = self.lvi_norm1(txt_tokens + txt_attn)
        txt_res = self.lvi_norm2(txt_res + self.lvi_ffn(txt_res))

        vis_query = vis_tokens + (vis_pos if vis_pos is not None else 0)
        vis_key = txt_res + (txt_pos if txt_pos is not None else 0)
        vis_out, attn_weights = self.vli_attn(vis_query, vis_key, vis_key, need_weights=True)
        vis_res = self.vli_norm1(vis_tokens + vis_out)
        vis_res = self.vli_norm2(vis_res + self.vli_ffn(vis_res))
        return txt_res, vis_res, attn_weights


class ID_MSCMA(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_levels: int = 4,
        n_points: int = 4,
        n_layers: int = 2,
        weight_sharing: bool = True,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.weight_sharing = weight_sharing
        if weight_sharing:
            self.layer = BiDirCMLayer(d_model, n_heads, n_levels, n_points)
        else:
            self.layer = nn.ModuleList(
                BiDirCMLayer(d_model, n_heads, n_levels, n_points) for _ in range(n_layers)
            )

    def forward(
        self,
        txt_tokens: torch.Tensor,
        feats: list[torch.Tensor],
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        reference_points: torch.Tensor,
        vis_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if vis_tokens is None:
            vis_tokens = torch.cat([feat.flatten(2).transpose(1, 2) for feat in feats], dim=1)

        txt_out, vis_out, attn = txt_tokens, vis_tokens, None
        for idx in range(self.n_layers):
            layer = self.layer if self.weight_sharing else self.layer[idx]
            txt_out, vis_out, attn = layer(
                txt_out,
                vis_out,
                feats,
                spatial_shapes,
                level_start_index,
                reference_points,
            )
        return txt_out, vis_out, attn


def attention_alignment_loss(
    attn_weights: torch.Tensor, gt_mask: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    if gt_mask.dim() == 1:
        gt_mask = gt_mask.unsqueeze(0)
    if attn_weights.dim() != 3:
        raise ValueError("Expected attention weights with shape [B, N, L].")

    if gt_mask.dim() == 2:
        gt_mask = gt_mask.unsqueeze(1)
    gt_mask = gt_mask.to(attn_weights.dtype)

    probs = attn_weights.mean(dim=2).unsqueeze(1).clamp(1e-6, 1 - 1e-6)
    if gt_mask.shape[-1] != probs.shape[-1]:
        raise ValueError("Ground truth mask and attention size mismatch.")
    return F.binary_cross_entropy(probs, gt_mask, reduction=reduction)

