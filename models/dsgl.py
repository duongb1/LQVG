import torch
import torch.nn as nn


class DSGL(nn.Module):
    def __init__(self, embed_dim: int = 256, heatmap_size: int = 32):
        super().__init__()
        self.heatmap_head = nn.Linear(embed_dim, heatmap_size * heatmap_size)
        self.heatmap_size = heatmap_size

        self.iou_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, obj_embeds: torch.Tensor):
        """Predict heatmaps and IoU scores from decoder object embeddings.

        Args:
            obj_embeds: Tensor of shape [B, N, C] containing decoder embeddings.

        Returns:
            A tuple ``(heatmap, iou_pred)`` where ``heatmap`` has shape
            [B, N, H, W] and ``iou_pred`` has shape [B, N, 1].
        """
        if obj_embeds.dim() != 3:
            raise ValueError(
                f"Expected obj_embeds to have shape [B, N, C], got {obj_embeds.shape}"
            )

        bsz, num_queries, _ = obj_embeds.shape

        heatmap = self.heatmap_head(obj_embeds)
        heatmap = heatmap.view(bsz, num_queries, self.heatmap_size, self.heatmap_size)
        heatmap = torch.sigmoid(heatmap)

        iou_pred = self.iou_head(obj_embeds)

        return heatmap, iou_pred
