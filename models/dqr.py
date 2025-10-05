import torch
from torch import nn


class DQR(nn.Module):
    """
    Dynamic Query Refiner:
    q_{l+1} = q_l + alpha * CA_text(q_l, T) + beta * CA_vis(q_l, V)
    - q: [Q,B,C], T: [L,B,C], V: [HW,B,C]
    """

    def __init__(self, d_model=256, nhead=8, attn_dropout=0.1,
                 alpha=0.2, beta=0.2, pre_ln=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pre_ln = pre_ln

        self.ca_text = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout, batch_first=False)
        self.ca_vis = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout, batch_first=False)

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_t = nn.LayerNorm(d_model)
        self.ln_v = nn.LayerNorm(d_model)

    @torch.no_grad()
    def set_coeff(self, alpha=None, beta=None):
        if alpha is not None:
            self.alpha = float(alpha)
        if beta is not None:
            self.beta = float(beta)

    def forward(self, q, text_tokens, visual_memory):
        # q: [Q,B,C], text_tokens: [L,B,C], visual_memory: [HW,B,C]
        if self.pre_ln:
            qn = self.ln_q(q)
            tn = self.ln_t(text_tokens)
            vn = self.ln_v(visual_memory)
        else:
            qn, tn, vn = q, text_tokens, visual_memory

        dT, _ = self.ca_text(qn, tn, tn, need_weights=False)  # [Q,B,C]
        dV, _ = self.ca_vis(qn, vn, vn, need_weights=False)  # [Q,B,C]
        return q + self.alpha * dT + self.beta * dV
