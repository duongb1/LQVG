# models/direction_parser.py
import re, torch, torch.nn.functional as F
from typing import List, Optional, Tuple

CornerPatterns = [
    (r"\b(upper|top)[-\s]*left\b|\btop[-\s]*left[-\s]*corner\b|\bupper[-\s]*left[-\s]*corner\b|\bnorth[-\s]*west\b|\bnorthwest\b", (-1, -1)),
    (r"\b(upper|top)[-\s]*right\b|\btop[-\s]*right[-\s]*corner\b|\bupper[-\s]*right[-\s]*corner\b|\bnorth[-\s]*east\b|\bnortheast\b", (1, -1)),
    (r"\b(lower|bottom)[-\s]*left\b|\bbottom[-\s]*left[-\s]*corner\b|\blower[-\s]*left[-\s]*corner\b|\bsouth[-\s]*west\b|\bsouthwest\b", (-1, 1)),
    (r"\b(lower|bottom)[-\s]*right\b|\bbottom[-\s]*right[-\s]*corner\b|\blower[-\s]*right[-\s]*corner\b|\bsouth[-\s]*east\b|\bsoutheast\b", (1, 1)),
]
SidePatterns = [
    (r"\b(far)?[-\s]*(left|west)\b|\b(left|west)[-\s]*most\b|\bfar(left|west)\b", (-1, 0)),
    (r"\b(far)?[-\s]*(right|east)\b|\b(right|east)[-\s]*most\b|\bfar(right|east)\b", (1, 0)),
    (r"\b(upper|top|north)\b|\btop[-\s]*side\b|\bupper[-\s]*part\b", (0, -1)),
    (r"\b(lower|bottom|south)\b|\bbottom[-\s]*side\b|\blower[-\s]*part\b", (0, 1)),
]
CenterPattern = r"\b(center|centre|middle)\b"

def _normalize(tokens: List[str]) -> str:
    txt = " ".join(tokens).lower()
    txt = re.sub(r"[_/]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def parse_direction_tokens(tokens: List[str]) -> Optional[Tuple[int, int]]:
    text = _normalize(tokens)
    for pat, vec in CornerPatterns:
        if re.search(pat, text): return vec
    for pat, vec in SidePatterns:
        if re.search(pat, text): return vec
    if re.search(CenterPattern, text): return (0, 0)
    return None

def make_direction_fields_multi(feats, vec, device):
    """
    feats: list các level [B,C,H,W]; vec=(dx,dy)
    return: list prior mỗi level [1,1,Hi*Wi] trong [0,1]
    """
    fields = []
    H0, W0 = feats[0].shape[-2:]
    ys, xs = torch.meshgrid(torch.linspace(0,1,H0,device=device),
                            torch.linspace(0,1,W0,device=device), indexing="ij")
    dx, dy = vec
    if dx == 0 and dy == 0:
        field0 = 1.0 - torch.clamp((xs-0.5).abs() + (ys-0.5).abs(), 0, 1)
    else:
        proj = dx*(0.5 - xs) + dy*(0.5 - ys)
        field0 = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
    field0 = field0.view(1,1,H0,W0)
    for f in feats:
        h,w = f.shape[-2:]
        f_lvl = F.interpolate(field0, size=(h,w), mode="bilinear", align_corners=False).reshape(1,1,h*w)
        fields.append(f_lvl)
    return fields
