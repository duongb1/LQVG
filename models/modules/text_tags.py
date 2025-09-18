import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyTagger(nn.Module):
    """
    Tagger nhẹ dựa trên embedding (không cần lexicon):
    đầu ra: prob BxLx5 theo thứ tự [color, size, shape, abs, rel]
    """

    def __init__(self, d_model=256, hidden=256, num_tags=5, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_tags),
        )

    def forward(self, text_tokens, text_padding_mask=None):
        """
        text_tokens: (B*T, L, d)
        text_padding_mask: (B*T, L) True=pad
        """
        logits = self.net(text_tokens)
        prob = logits.softmax(dim=-1)
        if text_padding_mask is not None:
            pad = text_padding_mask.unsqueeze(-1).float()
            prob = prob * (~text_padding_mask).unsqueeze(-1).float() + 0.0 * pad
        return prob
