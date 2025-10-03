# text_guide_linear.py
import torch
import torch.nn as nn
from typing import Optional


class GF_block(nn.Module):
    """
    Gated fusion khớp giữa đặc trưng thị giác & văn bản để sinh embedding điều kiện (1 token / 1 spatial cell).

    Tham số:
        d_model:     Kích thước kênh không gian trung gian để trộn (đầu ra của nhánh 'down' trước khi sinh scale).
        d_model_visual: Số kênh của đặc trưng thị giác (token hoặc fmap) trước khi 'down'.
        d_text:      Kích thước kênh của đặc trưng văn bản đưa vào.
        isTR:        True nếu x là token có dạng [B, L_v, C_v]; False nếu x là fmap [B, C_v, H, W].
        freeze_text: Nếu True sẽ detach word_feat_embed (điều kiện “raw text”, không backprop qua text).
    """
    def __init__(
        self,
        d_model: int = 256,
        d_model_visual: int = 256,
        d_text: int = 256,
        isTR: bool = True,
        freeze_text: bool = False,
    ):
        super().__init__()
        self.isTR = isTR
        self.freeze_text = freeze_text

        # Down proj cho nhánh thị giác (tuỳ token/fmap)
        if isTR:
            self.visual_down = nn.Linear(d_model_visual, d_model, bias=False)
        else:
            self.visual_down = nn.Conv2d(d_model_visual, d_model, kernel_size=1, bias=False)

        # Down proj cho nhánh văn bản
        self.text_down = nn.Linear(d_text, d_model, bias=False)

        # MLP sinh hệ số điều biến (lambda) theo thống kê của thị giác
        self.mm = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        word_feat_embed: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Input:
            x:               [B, L_v, C_v] nếu isTR=True, hoặc [B, C_v, H, W] nếu isTR=False
            word_feat_embed: [B, L_t, d_text] hoặc [B, 1, d_text]
            text_mask:       (tuỳ chọn) [B, L_t] với True là padding; nếu None sẽ dùng mean thường.

        Output:
            mm_embed:        [B, 1, d_model] — embedding điều kiện đa mô-đun (1 token)
        """
        if self.freeze_text:
            word_feat_embed = word_feat_embed.detach()

        # Text pooling (mask-aware nếu có)
        word = self.text_down(word_feat_embed)  # [B, L_t, d_model] hoặc [B, 1, d_model]
        if text_mask is not None:
            # True = padding → loại khỏi mean
            valid = (~text_mask).float().unsqueeze(-1)  # [B, L_t, 1]
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            word = (word * valid).sum(dim=1, keepdim=True) / denom  # [B, 1, d_model]
        else:
            word = word.mean(dim=1, keepdim=True)  # [B, 1, d_model]

        # Visual pooling tuỳ chế độ
        if self.isTR:
            # x: [B, L_v, C_v] -> down -> [B, L_v, d_model] -> mean L_v
            x_down = self.visual_down(x)
            x_mean = x_down.mean(dim=1, keepdim=True)  # [B, 1, d_model]
        else:
            # x: [B, C_v, H, W] -> down -> [B, d_model, H, W] -> mean spatial
            x_down = self.visual_down(x)
            x_mean = x_down.mean(dim=(2, 3), keepdim=True)  # [B, d_model, 1, 1]
            x_mean = x_mean.flatten(2).transpose(1, 2)      # [B, 1, d_model]

        # Gating
        lam = torch.sigmoid(self.mm(x_mean))  # [B, 1, d_model]
        mm_embed = word + lam * x_mean        # [B, 1, d_model]
        return mm_embed


class MM_adaption_linear(nn.Module):
    """
    Adapter điều biến ở mức token (sau khi đã thành chuỗi token).

    Thiết kế:
      x --(Linear down d_model_visual->mid)--> x_mid --(*)--> up -> Δx --(residual)--> y
                           ^                            ^
                           |                            |
                        gen_scale(cond)           Linear(mid->d_model_visual)
                           ^
                           |
                  cond = GF_block(x, word)

    Chú ý:
      - 'mid' nên tỷ lệ theo d_model_visual để tránh mismatch khi backbone output channel != hidden_dim.
      - gen_scale zero-init → near-identity ở bước đầu, ổn định train.
    """
    def __init__(
        self,
        d_model: int = 256,
        d_model_visual: int = 256,
        down_rate: int = 4,
        d_text: int = 256,
        freeze_text: bool = False,
    ):
        super().__init__()
        mid = max(1, d_model_visual // max(1, down_rate))
        self.down = nn.Linear(d_model_visual, mid, bias=False)
        self.fuse = GF_block(
            d_model=d_model,
            d_model_visual=d_model_visual,
            d_text=d_text,
            isTR=True,
            freeze_text=freeze_text,
        )
        self.gen_scale = nn.Linear(d_model, mid, bias=True)
        self.up = nn.Linear(mid, d_model_visual, bias=False)

        # near-identity init
        nn.init.zeros_(self.gen_scale.weight)
        nn.init.zeros_(self.gen_scale.bias)

    def forward(
        self,
        x: torch.Tensor,
        word_feat_embed: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:               [B, L, C_v]
        word_feat_embed: [B, L_t, d_text] hoặc [B, 1, d_text]
        text_mask:       (tuỳ chọn) [B, L_t] với True là padding
        """
        x_mid = self.down(x)                                 # [B, L, mid]
        cond = self.fuse(x, word_feat_embed, text_mask)      # [B, 1, d_model]
        scale = self.gen_scale(cond)                         # [B, 1, mid]
        x_mid = x_mid * scale                                # broadcast theo L
        x_delta = self.up(x_mid)                             # [B, L, C_v]
        return x + x_delta                                   # residual


class MM_adaption_conv_2d(nn.Module):
    """
    Adapter điều biến trên feature map 2D (sau input_proj).

    - Tự thích nghi số kênh thị giác khi thay đổi backbone/FPN (reconfigure modules nếu channel mismatch).
    - Giữ zero-init ở gen_scale để khởi tạo gần-identity.

    Forward:
        x: [B, C_v, H, W]
        word_feat_embed: [B, L_t, d_text] hoặc [B, 1, d_text]
        text_mask: (tuỳ chọn) [B, L_t] (True=padding)
    """
    def __init__(
        self,
        d_model: int = 256,
        d_model_visual: int = 256,
        down_rate: int = 4,
        d_text: int = 256,
        freeze_text: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_text = d_text
        self.freeze_text = freeze_text
        self.down_rate = max(1, down_rate)

        # mid theo kênh thị giác
        self.mid = max(1, d_model_visual // self.down_rate)

        # sinh scale từ cond (độ dài d_model → mid)
        self.gen_scale = nn.Linear(d_model, self.mid, bias=True)
        nn.init.zeros_(self.gen_scale.weight)
        nn.init.zeros_(self.gen_scale.bias)

        # modules phụ thuộc số kênh thị giác (set động)
        self._visual_channels = None
        # placeholders để mypy không kêu (được đặt thật ở _set_visual_dim)
        self.down: nn.Conv2d
        self.up: nn.Conv2d
        self.fuse: GF_block
        self._set_visual_dim(d_model_visual)

    def _set_visual_dim(self, channels: int) -> None:
        channels = int(channels)
        if self._visual_channels == channels:
            return
        self._visual_channels = channels
        self.down = nn.Conv2d(channels, self.mid, kernel_size=3, stride=1, padding=1, bias=False)
        self.up = nn.Conv2d(self.mid, channels, kernel_size=1, bias=False)
        self.fuse = GF_block(
            d_model=self.d_model,
            d_model_visual=channels,
            d_text=self.d_text,
            isTR=False,
            freeze_text=self.freeze_text,
        )

    def forward(
        self,
        x: torch.Tensor,
        word_feat_embed: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"MM_adaption_conv_2d expects 4D input, got {tuple(x.shape)}")

        # Thích nghi kênh nếu khác cấu hình hiện thời (ví dụ load checkpoint cũ hoặc backbone khác)
        visual_channels = x.shape[1]
        if self._visual_channels != visual_channels:
            self._set_visual_dim(visual_channels)
            # đồng bộ dtype/device cho các module vừa re-init
            self.down.to(device=x.device, dtype=x.dtype)
            self.up.to(device=x.device, dtype=x.dtype)
            self.fuse.to(device=x.device, dtype=x.dtype)

        x_mid = self.down(x)                                   # [B, mid, H, W]
        cond = self.fuse(x, word_feat_embed, text_mask)        # [B, 1, d_model]
        scale = self.gen_scale(cond).transpose(1, 2)           # [B, mid, 1]
        x_mid = x_mid * scale[..., None]                       # [B, mid, H, W]
        x_delta = self.up(x_mid)                               # [B, C_v, H, W]
        return x + x_delta                                     # residual
