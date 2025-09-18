import torch
import torch.nn.functional as F
from torch import nn

from models.id_mscma import ID_MSCMA, FiLMLite, attention_alignment_loss

import os
import math
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       nested_tensor_from_videos_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from util.losses import attention_alignment_loss

from .position_encoding import PositionEmbeddingSine1D
from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .segmentation import VisionLanguageFusionModule
from .matcher import build_matcher
from .criterion import SetCriterion
from .postprocessors import build_postprocessors
from .dsgl import DSGL

from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizerFast

import copy
from einops import rearrange, repeat


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)


class LQVG(nn.Module):

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 num_frames, aux_loss=False, with_box_refine=False, two_stage=False,
                 freeze_text_encoder=False, lambda_aal=0.0):

        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides[-3:])
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[-3:][_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):  # downsample 2x
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-3:][0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.num_frames = num_frames
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        assert two_stage == False, "args.two_stage must be false!"

        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        # self.tokenizer = RobertaTokenizerFast.from_pretrained('./weights/tokenizer')
        # self.text_encoder = RobertaModel.from_pretrained('./weights/text_encoder')
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        # resize the bert output channel to transformer d_model
        self.resizer = FeatureResizer(
            input_feat_size=768,
            output_feat_size=hidden_dim,
            dropout=0.1,
        )

        self.fusion_module = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8)
        self.fusion_module_text = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8)

        self.text_pos = PositionEmbeddingSine1D(hidden_dim, normalize=True)
        self.poolout_module = RobertaPoolout(d_model=hidden_dim)
        self.dsgl = DSGL(embed_dim=hidden_dim, heatmap_size=32)
        self.lambda_aal = lambda_aal
        self.id_mscma = ID_MSCMA(d_model=hidden_dim, n_heads=8, n_levels=4, n_points=4, n_layers=2, weight_sharing=True)
        self.film2 = FiLMLite(d_text=hidden_dim, c_vis=hidden_dim, init_alpha=0.1)
        self.film3 = FiLMLite(d_text=hidden_dim, c_vis=hidden_dim, init_alpha=0.1)
        self.lambda_aal = 0.05

    def forward(self, samples: NestedTensor, captions, targets):

        # Backbone
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list(samples)

        # features (list[NestedTensor]): res2 -> res5, shape of tensors is [B*T, Ci, Hi, Wi]
        # pos (list[Tensor]): shape of [B*T, C, Hi, Wi]
        features, pos = self.backbone(samples)

        b = len(captions)
        t = pos[0].shape[0] // b

        if 'valid_indices' in targets[0]:
            valid_indices = torch.tensor([i * t + target['valid_indices'] for i, target in enumerate(targets)]).to(
                pos[0].device)
            for feature in features:
                feature.tensors = feature.tensors.index_select(0, valid_indices)
                feature.mask = feature.mask.index_select(0, valid_indices)
            for i, p in enumerate(pos):
                pos[i] = p.index_select(0, valid_indices)
            samples.mask = samples.mask.index_select(0, valid_indices)
            for target in targets:
                if 'valid_indices' not in target:
                    continue
                vidx = target['valid_indices']
                if isinstance(vidx, torch.Tensor):
                    vidx = int(vidx.item())
                else:
                    vidx = int(vidx)
                if 'boxes' in target:
                    target['boxes'] = target['boxes'][vidx:vidx + 1]
                if 'valid' in target:
                    target['valid'] = target['valid'][vidx:vidx + 1]
            # t: num_frames -> 1
            t = 1

        text_features = self.forward_text(captions, device=pos[0].device)

        # prepare vision and text features for transformer
        srcs = []
        masks = []
        poses = []

        text_pos = self.text_pos(text_features).permute(2, 0, 1)  # [length, batch_size, c]
        text_word_features, text_word_masks = text_features.decompose()
        text_word_masks = text_word_masks.to(text_word_features.device)

        text_word_features = text_word_features.permute(1, 0, 2)  # [length, batch_size, c]
        text_word_initial_features = text_word_features

        attn_last = None
        mscma_token_mask = None
        mscma_hw = None
        attn_frames = t

        # Follow Deformable-DETR, we use the last three stages outputs from backbone
        for l, (feat, pos_l) in enumerate(zip(features[-3:], pos[-3:])):
            src, mask = feat.decompose()
            src_proj_l = self.input_proj[l](src)
            n, c, h, w = src_proj_l.shape

            # vision language early-fusion
            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> (t h w) b c', b=b, t=t)
            mask_tokens = rearrange(mask, '(b t) h w -> b (t h w)', b=b, t=t)
            pos_l = rearrange(pos_l, '(b t) c h w -> (t h w) b c', b=b, t=t)
            text_word_features, attn_weights = self.fusion_module_text(
                tgt=text_word_features,
                memory=src_proj_l,
                memory_key_padding_mask=mask_tokens,
                pos=pos_l,
                query_pos=None,
                need_weights=True,
            )
            attn_last = attn_weights
            mscma_token_mask = mask_tokens
            mscma_hw = (h, w)
            attn_frames = t

            src_proj_l = self.fusion_module(tgt=src_proj_l,
                                            memory=text_word_initial_features,
                                            memory_key_padding_mask=text_word_masks,
                                            pos=text_pos,
                                            query_pos=None)
            src_proj_l = rearrange(src_proj_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            mask = rearrange(mask_tokens, 'b (t h w) -> (b t) h w', t=t, h=h, w=w)
            pos_l = rearrange(pos_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            assert mask is not None

        if self.num_feature_levels > (len(features) - 1):
            _len_srcs = len(features) - 1  # fpn level
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                # vision language early-fusion
                src = rearrange(src, '(b t) c h w -> (t h w) b c', b=b, t=t)
                mask_tokens = rearrange(mask, '(b t) h w -> b (t h w)', b=b, t=t)
                pos_l = rearrange(pos_l, '(b t) c h w -> (t h w) b c', b=b, t=t)

                text_word_features, attn_weights = self.fusion_module_text(
                    tgt=text_word_features,
                    memory=src,
                    memory_key_padding_mask=mask_tokens,
                    pos=pos_l,
                    query_pos=None,
                    need_weights=True,
                )
                attn_last = attn_weights
                mscma_token_mask = mask_tokens
                mscma_hw = (h, w)
                attn_frames = t
                src = self.fusion_module(tgt=src,
                                         memory=text_word_initial_features,
                                         memory_key_padding_mask=text_word_masks,
                                         pos=text_pos,
                                         query_pos=None
                                         )
                src = rearrange(src, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
                mask = rearrange(mask_tokens, 'b (t h w) -> (b t) h w', t=t, h=h, w=w)
                pos_l = rearrange(pos_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)

        text_word_features = rearrange(text_word_features, 'l b c -> b l c')
        TXT_TOKENS = text_word_features
        dir_token_lists = []
        for caption in captions:
            if isinstance(caption, str):
                normalized = caption.lower().replace('/', ' ').replace('_', ' ')
                dir_token_lists.append(normalized.split())
            else:
                dir_token_lists.append([])
        feats = srcs
        t_global = TXT_TOKENS[:, 0]
        if t > 1:
            t_global_vis = t_global.unsqueeze(1).expand(b, t, -1).reshape(b * t, -1)
        else:
            t_global_vis = t_global
        if len(feats) > 1:
            feats[1] = self.film2(feats[1], t_global_vis)
        if len(feats) > 2:
            feats[2] = self.film3(feats[2], t_global_vis)

        spatial_shapes = torch.as_tensor([[f.shape[-2], f.shape[-1]] for f in feats],
                                         dtype=torch.long, device=feats[0].device)
        level_start_index = spatial_shapes.prod(1).cumsum(0)
        level_start_index = torch.roll(level_start_index, 1, 0)
        level_start_index[0] = 0
        txt_tokens_mscma = TXT_TOKENS
        dir_tokens = dir_token_lists
        if t > 1:
            txt_tokens_mscma = TXT_TOKENS.unsqueeze(1).expand(b, t, -1, -1).reshape(b * t, TXT_TOKENS.shape[1], TXT_TOKENS.shape[2])
            dir_tokens = [tokens.copy() for tokens in dir_token_lists for _ in range(t)]
        reference_points = torch.full((txt_tokens_mscma.shape[0], txt_tokens_mscma.shape[1], len(feats), 2), 0.5, device=feats[0].device)
        if len(dir_tokens) != txt_tokens_mscma.shape[0]:
            dir_tokens = None
        t2, v2, attn_last = self.id_mscma(
            txt_tokens_mscma, feats, spatial_shapes, level_start_index, reference_points,
            vis_tokens=None, dir_tokens=dir_tokens, anneal=(0.3, 0.1)
        )
        Ns = [(h * w).item() for h, w in spatial_shapes]
        splits = torch.split(v2, Ns, dim=1)
        refined_feats = [s.transpose(1, 2).reshape(txt_tokens_mscma.shape[0], self.hidden_dim, h.item(), w.item()) for s, (h, w) in zip(splits, spatial_shapes)]
        srcs = refined_feats
        if t > 1:
            text_word_features = t2.view(b, t, TXT_TOKENS.shape[1], TXT_TOKENS.shape[2]).mean(dim=1)
        else:
            text_word_features = t2
        text_sentence_features = self.poolout_module(text_word_features)

        # Transformer
        query_embeds = self.query_embed.weight  # [num_queries, c]
        text_embed = repeat(text_sentence_features, 'b c -> b t q c', t=t, q=self.num_queries)
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples = \
            self.transformer(srcs, text_embed, masks, poses, query_embeds)


        out = {}
        out['mscma_txt'] = t2
        out['mscma_vis'] = v2
        out['mscma_attn'] = attn_last
        # prediction
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()  # cxcywh, range in [0,1]
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        # rearrange
        outputs_class = rearrange(outputs_class, 'l (b t) q k -> l b t q k', b=b, t=t)
        outputs_coord = rearrange(outputs_coord, 'l (b t) q n -> l b t q n', b=b, t=t)
        out['pred_logits'] = outputs_class[-1]  # [batch_size, time, num_queries_per_frame, num_classes]
        out['pred_boxes'] = outputs_coord[-1]  # [batch_size, time, num_queries_per_frame, 4]

        last_layer_embeds = hs[-1]
        pred_heatmap, pred_iou = self.dsgl(last_layer_embeds)
        pred_heatmap = rearrange(pred_heatmap, '(b t) q h w -> b t q h w', b=b, t=t)
        pred_iou = rearrange(pred_iou.squeeze(-1), '(b t) q -> b t q', b=b, t=t)

        out['pred_heatmap'] = pred_heatmap
        out['pred_iou'] = pred_iou

        if (attn_last is not None and mscma_token_mask is not None and mscma_hw is not None
                and targets is not None and len(targets) > 0 and self.lambda_aal > 0):
            device = attn_last.device
            h_tokens, w_tokens = mscma_hw
            gt_masks = []
            for target in targets:
                target_boxes = target.get('boxes')
                target_valid = target.get('valid')
                if target_boxes is None or target_valid is None:
                    gt_masks.append(torch.zeros(attn_frames * h_tokens * w_tokens, device=device))
                    continue

                if target_boxes.dim() == 1:
                    target_boxes = target_boxes.unsqueeze(0)
                if target_valid.dim() == 0:
                    target_valid = target_valid.unsqueeze(0)

                if target_boxes.size(0) < attn_frames:
                    pad_boxes = target_boxes.new_zeros(attn_frames - target_boxes.size(0), target_boxes.size(1))
                    target_boxes = torch.cat([target_boxes, pad_boxes], dim=0)
                else:
                    target_boxes = target_boxes[:attn_frames]

                if target_valid.size(0) < attn_frames:
                    pad_valid = target_valid.new_zeros(attn_frames - target_valid.size(0))
                    target_valid = torch.cat([target_valid, pad_valid], dim=0)
                else:
                    target_valid = target_valid[:attn_frames]

                frame_masks = []
                for frame_idx in range(attn_frames):
                    if target_valid[frame_idx] <= 0:
                        frame_masks.append(torch.zeros(h_tokens, w_tokens, device=device))
                        continue

                    cx, cy, bw, bh = target_boxes[frame_idx]
                    x0 = (cx - bw / 2.0) * w_tokens
                    x1 = (cx + bw / 2.0) * w_tokens
                    y0 = (cy - bh / 2.0) * h_tokens
                    y1 = (cy + bh / 2.0) * h_tokens

                    x0 = int(torch.floor(x0).clamp(min=0, max=w_tokens - 1).item())
                    y0 = int(torch.floor(y0).clamp(min=0, max=h_tokens - 1).item())
                    x1 = int(torch.ceil(x1).clamp(min=0, max=w_tokens).item())
                    y1 = int(torch.ceil(y1).clamp(min=0, max=h_tokens).item())
                    x1 = max(x1, x0 + 1)
                    y1 = max(y1, y0 + 1)

                    mask_frame = torch.zeros(h_tokens, w_tokens, device=device)
                    mask_frame[y0:y1, x0:x1] = 1.0
                    frame_masks.append(mask_frame)

                frame_masks = torch.stack(frame_masks, dim=0)
                gt_masks.append(frame_masks.reshape(-1))

            gt_mask_flat = torch.stack(gt_masks, dim=0)
            valid_token_mask = (~mscma_token_mask).float()
            gt_mask_flat = gt_mask_flat * valid_token_mask

            text_valid = (~text_word_masks).float()
            valid_token_count = text_valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            attn_mean = (attn_last * text_valid.unsqueeze(-1)).sum(dim=1) / valid_token_count
            attn_mean = attn_mean.clamp(0.0, 1.0)
            gt_mask_flat = gt_mask_flat.clamp(0.0, 1.0)

            loss_aal = attention_alignment_loss(attn_mean, gt_mask_flat, reduction="mean")
            out['loss_aal'] = loss_aal * self.lambda_aal

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)
            text_attention_mask = tokenized.attention_mask.ne(1).bool()

            text_features = encoded_text.last_hidden_state
            text_features = self.resizer(text_features)
            text_masks = text_attention_mask
            text_features = NestedTensor(text_features, text_masks)  # NestedTensor
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return text_features


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class RobertaPoolout(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def build(args):
    if args.binary:
        num_classes = 1
    else:
        if args.dataset_file == 'ytvos':
            num_classes = 65
        elif args.dataset_file == 'davis':
            num_classes = 78
        elif args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb':
            num_classes = 1
        else:
            num_classes = 91  # for coco
    device = torch.device(args.device)

    # backbone
    if 'video_swin' in args.backbone:
        from .video_swin_transformer import build_video_swin_backbone
        backbone = build_video_swin_backbone(args)
    elif 'swin' in args.backbone:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(args)
    else:
        backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    model = LQVG(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        num_frames=args.num_frames,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        freeze_text_encoder=args.freeze_text_encoder,
        lambda_aal=args.lambda_aal
    )
    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_ce'] = args.cls_loss_coef
    weight_dict['loss_bbox'] = args.bbox_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_dsgl'] = 1.0
    weight_dict['loss_aal'] = 1.0
    if args.masks:  # always true
        weight_dict['loss_mask'] = args.mask_loss_coef
        weight_dict['loss_dice'] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'dsgl', 'aal']
    if args.masks:
        losses += ['masks']
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        focal_alpha=args.focal_alpha)
    criterion.to(device)

    # postprocessors, this is used for coco pretrain but not for rvos
    postprocessors = build_postprocessors(args, args.dataset_file)
    return model, criterion, postprocessors