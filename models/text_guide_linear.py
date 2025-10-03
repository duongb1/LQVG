import torch
import torch.nn as nn


class GF_block(nn.Module):
    def __init__(self, d_model=256, d_model_visual=256, d_text=256, isTR=True):
        """
        - isTR=True: x là token [B, L_v, C_v]
        - isTR=False: x là fmap [B, C_v, H, W]
        """
        super().__init__()
        self.isTR = isTR
        if isTR:
            self.visual_down = nn.Linear(d_model_visual, d_model, bias=False)
        else:
            self.visual_down = nn.Conv2d(d_model_visual, d_model, kernel_size=1, bias=False)
        self.text_down = nn.Linear(d_text, d_model, bias=False)
        self.mm = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
        )

    def forward(self, x, word_feat_embed):
        # word_feat_embed: [B, L_t, d_text] hoặc [B, 1, d_text]
        word = self.text_down(word_feat_embed)          # [B, L_t, d_model]
        word = word.mean(dim=1, keepdim=True)           # [B, 1, d_model]

        if self.isTR:
            # x: [B, L_v, C_v] -> [B, L_v, d_model]
            x = self.visual_down(x)
            x_mean = x.mean(dim=1, keepdim=True)        # [B, 1, d_model]
        else:
            # x: [B, C_v, H, W] -> [B, d_model, H, W]
            x = self.visual_down(x)
            x_mean = x.mean(dim=(2, 3), keepdim=True)   # [B, d_model, 1, 1]
            x_mean = x_mean.flatten(2).transpose(1, 2)  # [B, 1, d_model]

        lam = torch.sigmoid(self.mm(x_mean))            # [B, 1, d_model]
        mm_embed = word + lam * x_mean                  # [B, 1, d_model]
        return mm_embed


class MM_adaption_linear(nn.Module):
    def __init__(self, d_model=256, d_model_visual=256, down_rate=4, d_text=256):
        """
        Điều biến token-level (batch-first):
          x: [B, L, C_v], word_feat_embed: [B, L_t, d_text]
        """
        super().__init__()
        mid = max(1, d_model // down_rate)
        self.down = nn.Linear(d_model_visual, mid, bias=False)
        self.fuse = GF_block(d_model=d_model, d_model_visual=d_model_visual, d_text=d_text, isTR=True)
        self.gen_scale = nn.Linear(d_model, mid, bias=True)
        self.up = nn.Linear(mid, d_model_visual, bias=False)
        # near-identity init (Δ≈0)
        nn.init.zeros_(self.gen_scale.weight)
        nn.init.zeros_(self.gen_scale.bias)

    def forward(self, x, word_feat_embed):
        # x: [B, L, C_v]
        x_mid = self.down(x)                            # [B, L, mid]
        cond = self.fuse(x, word_feat_embed)            # [B, 1, d_model]
        scale = self.gen_scale(cond)                    # [B, 1, mid]
        x_mid = x_mid * scale                           # broadcast theo L
        x_delta = self.up(x_mid)                        # [B, L, C_v]
        return x + x_delta                              # residual


class MM_adaption_conv_2d(nn.Module):
    def __init__(self, d_model=256, d_model_visual=256, down_rate=4, d_text=256):
        """
        Điều biến feature map trước input_proj:
          x: [B, C_v, H, W], word_feat_embed: [B, L_t, d_text]
        """
        super().__init__()
        mid = max(1, d_model // down_rate)
        self.down = nn.Conv2d(d_model_visual, mid, kernel_size=3, stride=1, padding=1, bias=False)
        self.fuse = GF_block(d_model=d_model, d_model_visual=d_model_visual, d_text=d_text, isTR=False)
        self.gen_scale = nn.Linear(d_model, mid, bias=True)
        self.up = nn.Conv2d(mid, d_model_visual, kernel_size=1, bias=False)
        # near-identity init
        nn.init.zeros_(self.gen_scale.weight)
        nn.init.zeros_(self.gen_scale.bias)

    def forward(self, x, word_feat_embed):
        B, _, H, W = x.shape
        x_mid = self.down(x)                            # [B, mid, H, W]
        cond = self.fuse(x, word_feat_embed)            # [B, 1, d_model]
        scale = self.gen_scale(cond).transpose(1, 2)    # [B, mid, 1]
        x_mid = x_mid * scale[..., None]                # [B, mid, H, W]
        x_delta = self.up(x_mid)                        # [B, C_v, H, W]
        return x + x_delta                              # residual
