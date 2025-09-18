import torch
import torch.nn as nn
from .text_tags import TinyTagger
from .abp import AttributeBranch
from .stp import STP


class PreMSCMAFusion(nn.Module):
    """
    Tiền xử lý trước MSCMA-Deform:
      1) Tagger gán trọng số token theo loại
      2) ABP (color/size/shape) FiLM lên ảnh
      3) STP (abs/rel) tạo prior vị trí + refine ảnh
    """

    def __init__(self, d_model=256, k_prompts=6, n_heads=8, dropout=0.0):
        super().__init__()
        self.tagger = TinyTagger(d_model=d_model, hidden=d_model, num_tags=5, dropout=dropout)
        self.abp = AttributeBranch(d_model=d_model, dropout=dropout)
        self.stp = STP(d_model=d_model, k_prompts=k_prompts, n_heads=n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, vis_tokens, text_tokens, text_padding_mask=None):
        """
        vis_tokens: (B*T, Nv, d)
        text_tokens: (B*T, L, d)
        text_padding_mask: (B*T, L) True=pad
        """
        tag_prob = self.tagger(text_tokens, text_padding_mask)
        vis1, abp_extra = self.abp(vis_tokens, text_tokens, tag_prob)
        vis2, prior, stp_extra = self.stp(vis1, text_tokens, tag_prob)
        out = self.norm(vis2)
        extras = {"tag_prob": tag_prob, **abp_extra, **stp_extra, "prior_score": prior}
        return out, extras
