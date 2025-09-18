import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeBranch(nn.Module):
    """
    FiLM/gating theo 3 nhóm thuộc tính: color, size, shape
    """

    def __init__(self, d_model=256, dropout=0.0):
        super().__init__()
        self.film_color = nn.Linear(d_model, 2 * d_model)
        self.film_size = nn.Linear(d_model, 2 * d_model)
        self.film_shape = nn.Linear(d_model, 2 * d_model)
        self.pool = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _weighted_pool(text_tokens, weights, eps=1e-6):
        w = weights.clamp(min=0).unsqueeze(-1)
        s = (text_tokens * w).sum(dim=1) / (w.sum(dim=1) + eps)
        return s

    @staticmethod
    def _apply_film(vis_tokens, vec, film_layer):
        gamma, beta = film_layer(vec).chunk(2, dim=-1)
        return vis_tokens * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def forward(self, vis_tokens, text_tokens, tag_prob):
        """
        vis_tokens: (B*T, Nv, d)
        text_tokens: (B*T, L, d)
        tag_prob: (B*T, L, 5) -> [color, size, shape, abs, rel]
        """
        p_color = tag_prob[..., 0]
        p_size = tag_prob[..., 1]
        p_shape = tag_prob[..., 2]

        t_color = self.pool(self._weighted_pool(text_tokens, p_color))
        t_size = self.pool(self._weighted_pool(text_tokens, p_size))
        t_shape = self.pool(self._weighted_pool(text_tokens, p_shape))

        out = vis_tokens
        out = self._apply_film(out, t_color, self.film_color)
        out = self._apply_film(out, t_size, self.film_size)
        out = self._apply_film(out, t_shape, self.film_shape)
        out = self.norm(out)
        out = self.drop(out)

        extras = {"t_color": t_color, "t_size": t_size, "t_shape": t_shape}
        return out, extras
