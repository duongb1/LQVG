import torch.nn as nn

from models.ops.modules import MSDeformAttn


class DeformLVI(nn.Module):
    """Text tokens query multi-scale visual features via Multi-Scale Deformable Attention."""

    def __init__(self, d_model=256, n_heads=8, n_levels=3, n_points=4, dropout=0.1):
        super().__init__()
        self.n_levels = n_levels
        self.ref_proj = nn.Linear(d_model, n_levels * 2)
        self.attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(
        self,
        text_tokens,
        vis_value,
        spatial_shapes,
        level_start_index,
        vis_key_padding_mask=None,
    ):
        """Run deformable LVI attention."""

        ref = self.ref_proj(text_tokens).view(text_tokens.shape[0], text_tokens.shape[1], self.n_levels, 2).sigmoid()
        ctx, _, _ = self.attn(text_tokens, ref, vis_value, spatial_shapes, level_start_index, vis_key_padding_mask)
        text_out = self.norm(text_tokens + self.out(ctx))
        text_out = self.norm_ffn(text_out + self.ffn(text_out))
        return text_out


class VLI(nn.Module):
    """Visual tokens query enriched text tokens via standard MHA."""

    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, vis_tokens, text_tokens, text_padding_mask=None):
        ctx, _ = self.cross(vis_tokens, text_tokens, text_tokens, key_padding_mask=text_padding_mask)
        vis_out = self.norm(vis_tokens + self.out(ctx))
        vis_out = self.norm_ffn(vis_out + self.ffn(vis_out))
        return vis_out


class MSCMADeformBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=3, n_points=4, dropout=0.1):
        super().__init__()
        self.lvi = DeformLVI(d_model, n_heads, n_levels, n_points, dropout)
        self.vli = VLI(d_model, n_heads, dropout)

    def forward(
        self,
        vis_tokens,
        text_tokens,
        vis_value,
        spatial_shapes,
        level_start_index,
        text_padding_mask=None,
        vis_key_padding_mask=None,
    ):
        text_out = self.lvi(text_tokens, vis_value, spatial_shapes, level_start_index, vis_key_padding_mask)
        vis_out = self.vli(vis_tokens, text_out, text_padding_mask)
        return vis_out, text_out


class MSCMADeform(nn.Module):
    """Two-stage LVI→VLI stack."""

    def __init__(self, d_model=256, n_heads=8, n_levels=3, n_points=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [MSCMADeformBlock(d_model, n_heads, n_levels, n_points, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        vis_tokens,
        text_tokens,
        vis_value,
        spatial_shapes,
        level_start_index,
        text_padding_mask=None,
        vis_key_padding_mask=None,
    ):
        v, t = vis_tokens, text_tokens
        for layer in self.layers:
            v, t = layer(v, t, vis_value, spatial_shapes, level_start_index, text_padding_mask, vis_key_padding_mask)
        return v, t
