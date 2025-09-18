import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .segmentation import (dice_loss, sigmoid_focal_loss)

from einops import rearrange

class SetCriterion(nn.Module):
    """ This class computes the loss for ReferFormer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.focal_alpha = focal_alpha
        self.mask_out_stride = 4
        self.lambda_heatmap = 2.0
        self.lambda_iou = 1.0

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] 
        _, nf, nq = src_logits.shape[:3]
        src_logits = rearrange(src_logits, 'b t q k -> b (t q) k')

        # judge the valid frames
        valid_indices = []
        valids = [target['valid'] for target in targets]
        for valid, (indice_i, indice_j) in zip(valids, indices): 
            valid_ind = valid.nonzero().flatten() 
            valid_i = valid_ind * nq + indice_i
            valid_j = valid_ind + indice_j * nf
            valid_indices.append((valid_i, valid_j))

        idx = self._get_src_permutation_idx(valid_indices) # NOTE: use valid indices
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, valid_indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device) 
        if self.num_classes == 1: # binary referred
            target_classes[idx] = 0
        else:
            target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            pass
        return losses


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']  
        bs, nf, nq = src_boxes.shape[:3]
        src_boxes = src_boxes.transpose(1, 2)  

        idx = self._get_src_permutation_idx(indices)
        src_boxes = src_boxes[idx]  
        src_boxes = src_boxes.flatten(0, 1)  # [b*t, 4]

        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)  # [b*t, 4]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses


    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        # tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"] 
        src_masks = src_masks.transpose(1, 2) 

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets], 
                                                              size_divisibility=32, split=False).decompose()
        target_masks = target_masks.to(src_masks) 

        # downsample ground truth masks with ratio mask_out_stride
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w

        src_masks = src_masks[src_idx] 
        # upsample predictions to the target size
        # src_masks = interpolate(src_masks, size=target_masks.shape[-2:], mode="bilinear", align_corners=False) 
        src_masks = src_masks.flatten(1) # [b, thw]

        target_masks = target_masks.flatten(1) # [b, thw]

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_dsgl(self, outputs, targets, indices, num_boxes):
        if 'pred_heatmap' not in outputs or 'pred_iou' not in outputs:
            device = next(iter(outputs.values())).device
            return {'loss_dsgl': torch.zeros([], device=device)}

        src_heatmap = outputs['pred_heatmap'].transpose(1, 2)
        src_iou = outputs['pred_iou'].transpose(1, 2)
        idx = self._get_src_permutation_idx(indices)

        if src_heatmap.numel() == 0:
            zero = src_heatmap.sum()
            return {'loss_dsgl': zero}

        batch_idx, _ = idx
        src_heatmap = src_heatmap[idx].reshape(-1, *src_heatmap.shape[-2:])
        src_iou = src_iou[idx].reshape(-1)

        target_heatmap = torch.stack([t['heatmap'] for t in targets], dim=0).to(src_heatmap.device)
        target_heatmap = target_heatmap[batch_idx].reshape(-1, *target_heatmap.shape[-2:])

        target_valid = torch.stack([t['valid'] for t in targets], dim=0).to(src_iou.device)
        target_valid = target_valid[batch_idx].reshape(-1).bool()

        pred_boxes = outputs['pred_boxes'].transpose(1, 2)[idx].reshape(-1, 4)
        target_boxes = torch.stack([t['boxes'] for t in targets], dim=0).to(pred_boxes.device)
        target_boxes = target_boxes[batch_idx].reshape(-1, 4)

        if target_valid.any():
            heatmap_loss = F.mse_loss(src_heatmap[target_valid], target_heatmap[target_valid])

            pred_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes[target_valid])
            target_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes[target_valid])
            iou_targets = box_ops.clip_iou(pred_boxes_xyxy, target_boxes_xyxy).detach()
            iou_loss = F.mse_loss(src_iou[target_valid], iou_targets)
        else:
            heatmap_loss = src_heatmap.sum() * 0
            iou_loss = src_iou.sum() * 0

        loss_dsgl = self.lambda_heatmap * heatmap_loss + self.lambda_iou * iou_loss
        return {'loss_dsgl': loss_dsgl}

    def loss_aal(self, outputs, targets, indices, num_boxes):

        logits = outputs.get('aal_logits')
        if logits is None:
            device = next(iter(outputs.values())).device
            return {'loss_aal': torch.zeros([], device=device)}


        gt_mask = outputs.get('aal_gt_mask')
        if gt_mask is None:
            spatial_shapes = outputs.get('mscma_shapes')
            if spatial_shapes is None:

                device = logits.device
                return {'loss_aal': torch.zeros([], device=device)}
            gt_mask = self._build_aal_mask_from_targets(targets, spatial_shapes, logits.device)
        else:
            gt_mask = gt_mask.to(logits.device)

        gt_mask = gt_mask.float()
        if gt_mask.shape != logits.shape:
            raise ValueError('Ground truth mask shape must match AAL logits')

        pos = gt_mask.sum().clamp(min=1.0)
        neg = (gt_mask.numel() - pos).clamp(min=1.0)
        pos_weight = (neg / pos).detach()

        loss = F.binary_cross_entropy_with_logits(
            logits,
            gt_mask,
            pos_weight=pos_weight,
            reduction='mean',
        )

        return {'loss_aal': loss}

    def _build_aal_mask_from_targets(self, targets, spatial_shapes, device):
        if isinstance(spatial_shapes, torch.Tensor):
            shapes = spatial_shapes.tolist()
        else:
            shapes = spatial_shapes

        total_tokens = sum(h * w for h, w in shapes)
        gt_mask = torch.zeros(len(targets), total_tokens, device=device)

        offset = 0
        for h, w in shapes:
            level_masks = []
            for target in targets:
                if 'masks' not in target or target['masks'] is None:
                    level_masks.append(torch.zeros(h * w, device=device))
                    continue

                tgt_masks = target['masks']
                if tgt_masks.dim() == 4:
                    tgt_masks = tgt_masks[:, 0]
                tgt_masks = tgt_masks.to(device=device, dtype=torch.float32)
                if tgt_masks.numel() == 0:
                    level_masks.append(torch.zeros(h * w, device=device))
                    continue

                valid = target.get('valid')
                if valid is not None:
                    valid = valid.to(device=device, dtype=torch.float32)
                    if valid.dim() == 1:
                        valid = valid.view(-1, 1, 1)
                    tgt_masks = tgt_masks[:valid.shape[0]] * valid[:tgt_masks.shape[0]]

                union_mask = tgt_masks.max(dim=0).values if tgt_masks.dim() > 2 else tgt_masks
                resized = F.interpolate(
                    union_mask.unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                level_masks.append(resized.flatten())

            if level_masks:
                gt_mask[:, offset:offset + h * w] = torch.stack(level_masks, dim=0)
            offset += h * w

        return gt_mask.clamp(0, 1)


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'dsgl': self.loss_dsgl,
            'aal': self.loss_aal
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        target_valid = torch.stack([t["valid"] for t in targets], dim=0).reshape(-1) # [B, T] -> [B*T]
        num_boxes = target_valid.sum().item() 
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


