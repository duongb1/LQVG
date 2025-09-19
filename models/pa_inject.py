import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_identity_linear_(lin: nn.Linear, gain=1e-3, bias_init=0.0):
    nn.init.zeros_(lin.weight)
    if lin.bias is not None:
        nn.init.constant_(lin.bias, bias_init)


class ChannelGate(nn.Module):
    """
    Sinh W_c \in R^C từ text vec: MLP(text) -> sigmoid (scale ~ [0,2]) ~ gần 1.0
    """

    def __init__(self, d_txt: int, C: int, hidden: int = 256):
        super().__init__()
        self.proj1 = nn.Linear(d_txt, hidden)
        self.act = nn.ReLU(inplace=True)
        self.proj2 = nn.Linear(hidden, C)
        _init_identity_linear_(self.proj2, gain=1e-3, bias_init=0.0)

    def forward(self, t_vec: torch.Tensor):
        # t_vec: [B, d_txt]
        x = self.proj1(t_vec)
        x = self.act(x)
        x = self.proj2(x)  # [B, C]
        g = torch.sigmoid(x) * 2.0   # ~[0,2], trung bình ~1.0
        return g  # [B,C]


class SpatialGate(nn.Module):
    """
    Sinh W_s \in R^{H x W} từ text vec:
    text -> seed [B,Cs,7,7] -> deconv/upsample -> conv1x1 -> sigmoid map
    """

    def __init__(self, d_txt: int, C_seed: int = 64, seed_hw: int = 7):
        super().__init__()
        self.seed_hw = seed_hw
        self.fc = nn.Linear(d_txt, C_seed * seed_hw * seed_hw)
        self.bn = nn.BatchNorm2d(C_seed)
        self.deconv = nn.ConvTranspose2d(
            C_seed, C_seed, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv1x1 = nn.Conv2d(C_seed, 1, kernel_size=1)
        # init gần identity
        _init_identity_linear_(self.fc, gain=1e-3, bias_init=0.0)
        nn.init.kaiming_uniform_(self.deconv.weight, a=1)
        nn.init.zeros_(self.deconv.bias)
        nn.init.zeros_(self.conv1x1.weight)
        nn.init.zeros_(self.conv1x1.bias)

    def forward(self, t_vec: torch.Tensor, H: int, W: int):
        B = t_vec.size(0)
        seed = self.fc(t_vec).view(B, -1, self.seed_hw, self.seed_hw)  # [B,Cs,7,7]
        seed = self.bn(seed)
        x = F.relu(seed, inplace=True)
        # upsample cho tới khi >= (H,W)
        while x.size(-2) < H or x.size(-1) < W:
            x = self.deconv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        m = torch.sigmoid(self.conv1x1(x))  # [B,1,H,W]
        return m


class BiasGate(nn.Module):
    """
    Sinh B \in R^C từ text vec: Linear(text) ~ 0 lúc đầu
    """

    def __init__(self, d_txt: int, C: int):
        super().__init__()
        self.proj = nn.Linear(d_txt, C)
        _init_identity_linear_(self.proj, gain=1e-3, bias_init=0.0)

    def forward(self, t_vec: torch.Tensor):
        return self.proj(t_vec)  # [B,C]


class PAStage(nn.Module):
    """
    PA cho một scale: F' = W_s * (W_c ⊙ F) + B
    with_spatial/bias cho phép cấu hình khác nhau theo scale.
    """

    def __init__(self, C: int, d_txt: int, with_spatial: bool, with_bias: bool,
                 ch_hidden: int = 256, sp_seed: int = 64):
        super().__init__()
        self.chan = ChannelGate(d_txt, C, hidden=ch_hidden)
        self.spa = SpatialGate(d_txt, C_seed=sp_seed, seed_hw=7) if with_spatial else None
        self.bias = BiasGate(d_txt, C) if with_bias else None

    def forward(self, F: torch.Tensor, t_vec: torch.Tensor):
        """
        F: [B,C,H,W], t_vec: [B,d_txt]
        """
        B, C, H, W = F.shape
        Wc = self.chan(t_vec).view(B, C, 1, 1)           # [B,C,1,1] ~ around 1
        out = F * Wc
        if self.spa is not None:
            Ws = self.spa(t_vec, H, W)                   # [B,1,H,W]
            out = out * Ws
        if self.bias is not None:
            Bv = self.bias(t_vec).view(B, C, 1, 1)       # [B,C,1,1]
            out = out + Bv
        return out.contiguous()


class PAInjector(nn.Module):
    """
    Gắn PA vào 3-4 scale ResNet (C3/C4/C5/C6).
    text_feats: dict chứa các vector ngôn ngữ đã chọn (thường là [CLS])
       keys mặc định: {'c3','c4','c5','c6'} -> [B,d_txt]
    """

    def __init__(self, chans: dict, d_txt: int,
                 use_c3=True, use_c4=True, use_c5=True, use_c6=True,
                 c3_spatial=False, c3_bias=False,
                 c4_spatial=True,  c4_bias=False,
                 c5_spatial=True,  c5_bias=False,
                 c6_spatial=False, c6_bias=True,
                 ch_hidden=256, sp_seed=64):
        super().__init__()
        self.use = {'c3': use_c3, 'c4': use_c4, 'c5': use_c5, 'c6': use_c6}
        self.pa = nn.ModuleDict()
        for k, C in chans.items():
            if not self.use.get(k, False):
                continue
            if k == 'c3':
                self.pa[k] = PAStage(C, d_txt, with_spatial=c3_spatial, with_bias=c3_bias,
                                     ch_hidden=ch_hidden, sp_seed=sp_seed)
            elif k == 'c4':
                self.pa[k] = PAStage(C, d_txt, with_spatial=c4_spatial, with_bias=c4_bias,
                                     ch_hidden=ch_hidden, sp_seed=sp_seed)
            elif k == 'c5':
                self.pa[k] = PAStage(C, d_txt, with_spatial=c5_spatial, with_bias=c5_bias,
                                     ch_hidden=ch_hidden, sp_seed=sp_seed)
            elif k == 'c6':
                self.pa[k] = PAStage(C, d_txt, with_spatial=c6_spatial, with_bias=c6_bias,
                                     ch_hidden=ch_hidden, sp_seed=sp_seed)

    def forward(self, feats: dict, text_feats: dict):
        """
        feats: {'c3': [B,C3,H3,W3], 'c4':..., 'c5':..., 'c6':...}
        text_feats: {'c3': [B,d], 'c4':[B,d], 'c5':[B,d], 'c6':[B,d]}
          -> nếu không có key nào, sẽ fallback sang 'cls'
        """
        out = {}
        cls_vec = text_feats.get('cls', None)
        for k, F in feats.items():
            if k not in self.pa:
                out[k] = F
                continue
            t_vec = text_feats.get(k, cls_vec)
            if t_vec is None:
                raise ValueError(f"PAInjector: missing text vector for {k} and no 'cls' provided.")
            out[k] = self.pa[k](F, t_vec)
        return out
