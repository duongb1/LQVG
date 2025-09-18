import torch
import torch.nn as nn
import torch.nn.functional as F


class STP(nn.Module):
    """
    Spatial-Text Prompting:
    - Từ text (abs+rel) sinh K prompts
    - MHA vis<->prompts để lấy prior và refine ảnh
    """

    def __init__(self, d_model=256, k_prompts=6, n_heads=8, dropout=0.0):
        super().__init__()
        self.k = k_prompts
        self.gen = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, k_prompts * d_model),
        )
        self.pos_prompt = nn.Parameter(torch.randn(k_prompts, d_model))
        self.cross = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _weighted_pool(text_tokens, weights, eps=1e-6):
        w = weights.clamp(min=0).unsqueeze(-1)
        s = (text_tokens * w).sum(dim=1) / (w.sum(dim=1) + eps)
        return s

    def forward(self, vis_tokens, text_tokens, tag_prob):
        """
        vis_tokens: (B*T, Nv, d)
        text_tokens: (B*T, L, d)
        tag_prob: (B*T, L, 5) -> [color, size, shape, abs, rel]
        """
        BxT, Nv, d = vis_tokens.shape
        p_abs = tag_prob[..., 3]
        p_rel = tag_prob[..., 4]

        t_abs = self._weighted_pool(text_tokens, p_abs)
        t_rel = self._weighted_pool(text_tokens, p_rel)
        t_seed = 0.7 * t_abs + 0.3 * t_rel

        prompts = self.gen(t_seed).view(BxT, self.k, d) + self.pos_prompt.unsqueeze(0)

        ctx, attn = self.cross(vis_tokens, prompts, prompts)
        out = self.norm(vis_tokens + ctx)
        prior = attn.mean(dim=-1)
        extras = {"prompts": prompts, "stp_attn": attn, "stp_prior": prior}
        return out, prior, extras
