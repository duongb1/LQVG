import torch
import torch.nn.functional as F
from torch import nn

import os
import math
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       nested_tensor_from_videos_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .position_encoding import PositionEmbeddingSine1D
from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
# NOTE: KHÔNG dùng bản cũ trong segmentation nữa
# from .segmentation import VisionLanguageFusionModule
from .matcher import build_matcher
from .criterion import SetCriterion
from .postprocessors import build_postprocessors

from transformers import RobertaModel, RobertaTokenizerFast

import copy
from einops import rearrange, repeat


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable HF tokenizer warning


# === NEW: Vision-Language Fusion block (MHA + FFN), pre-norm, residual ===
class VisionLanguageFusionModule(nn.Module):
    """
    Drop-in replacement cho module cũ:
      - Pre-norm: LN -> MHA -> resid; LN -> FFN -> resid
      - FFN: d -> 4d -> d
      - API giữ nguyên: forward(tgt, memory, memory_key_padding_mask=None, pos=None, query_pos=None)
    """
    def __init__(self, d_model, nhead, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _with_pos(x, pos):
        return x if pos is None else x + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None):
        # tgt, memory shape: [L, B, C] (đúng như bạn đang dùng)
        x = self.norm1(tgt)
        q = self._with_pos(x, query_pos)
        k = self._with_pos(memory, pos)
        v = memory
        attn_out = self.mha(q, k, v, key_padding_mask=memory_key_padding_mask)[0]
        x = tgt + self.drop(attn_out)

        y = self.norm2(x)
        y = self.ffn(y)
        x = x + self.drop(y)
        return x


class LQVG(nn.Module):

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 num_frames, aux_loss=False, with_box_refine=False, two_stage=False,
                 freeze_text_encoder=False):

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
            for _ in range(num_feature_levels - num_backbone_outs):
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
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        # text encoder
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        # resize bert output to hidden_dim
        self.resizer = FeatureResizer(
            input_feat_size=768,
            output_feat_size=hidden_dim,
            dropout=0.1,
        )

        # === Fusion modules ===
        # LVI (text <- vision) và VLI stack (vision <- text) đều dùng block mới (MHA+FFN)
        self.fusion_module_text = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8, dropout=0.1, ffn_mult=4)

        # === Progressive 3-layer VLI stack (independent modules) ===
        base_vli = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8, dropout=0.1, ffn_mult=4)
        self.vli_layers = nn.ModuleList([copy.deepcopy(base_vli) for _ in range(3)])
        self.vli_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        self.vli_res_scale = 0.7

        self.text_pos = PositionEmbeddingSine1D(hidden_dim, normalize=True)
        self.poolout_module = RobertaPoolout(d_model=hidden_dim)

        # optional debug
        if os.environ.get("LQVG_DEBUG", "0") == "1":
            self._debug_vli_once()

    def _debug_vli_once(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        vli_total = sum(p.numel() for n, p in self.named_parameters() if "vli_layers" in n)
        print(f"\n[LQVG_DEBUG] Trainable total: {total} (= {total/1e6:.3f}M)")
        print(f"[LQVG_DEBUG] VLI total:       {vli_total} (= {vli_total/1e6:.3f}M)")
        for i, lyr in enumerate(self.vli_layers):
            cnt = sum(p.numel() for p in lyr.parameters() if p.requires_grad)
            print(f"  - vli_layers[{i}]: {cnt} (= {cnt/1e6:.3f}M)")
        try:
            p0 = next(self.vli_layers[0].parameters())
            p1 = next(self.vli_layers[1].parameters())
            print(f"[LQVG_DEBUG] param id L0={id(p0)} L1={id(p1)} (diff -> OK)")
        except StopIteration:
            print("[LQVG_DEBUG] VLI has no params?")

    def forward(self, samples: NestedTensor, captions, targets):

        # Backbone
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list(samples)
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
            t = 1

        text_features = self.forward_text(captions, device=pos[0].device)

        srcs, masks, poses = [], [], []
        text_pos = self.text_pos(text_features).permute(2, 0, 1)
        text_word_features, text_word_masks = text_features.decompose()
        text_word_features = text_word_features.permute(1, 0, 2)
        text_word_initial_features = text_word_features

        # === multi-scale features ===
        for l, (feat, pos_l) in enumerate(zip(features[-3:], pos[-3:])):
            src, mask = feat.decompose()
            src_proj_l = self.input_proj[l](src)
            n, c, h, w = src_proj_l.shape

            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> (t h w) b c', b=b, t=t)
            mask = rearrange(mask, '(b t) h w -> b (t h w)', b=b, t=t)
            pos_l = rearrange(pos_l, '(b t) c h w -> (t h w) b c', b=b, t=t)

            # LVI: text <- vision
            text_word_features = self.fusion_module_text(
                tgt=text_word_features,
                memory=src_proj_l,
                memory_key_padding_mask=mask,
                pos=pos_l,
                query_pos=None)

            # === Progressive 3-layer VLI === (vision <- text)
            v_cur = src_proj_l
            for i in range(3):
                v_in = v_cur
                v_out = self.vli_layers[i](
                    tgt=self.vli_norms[i](v_in),
                    memory=text_word_initial_features,
                    memory_key_padding_mask=text_word_masks,
                    pos=text_pos,
                    query_pos=None
                )
                v_cur = v_in + self.vli_res_scale * v_out
            src_proj_l = v_cur

            src_proj_l = rearrange(src_proj_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            mask = rearrange(mask, 'b (t h w) -> (b t) h w', t=t, h=h, w=w)
            pos_l = rearrange(pos_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)

        # === FPN extra levels ===
        if self.num_feature_levels > (len(features) - 1):
            _len_srcs = len(features) - 1
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                src = rearrange(src, '(b t) c h w -> (t h w) b c', b=b, t=t)
                mask = rearrange(mask, '(b t) h w -> b (t h w)', b=b, t=t)
                pos_l = rearrange(pos_l, '(b t) c h w -> (t h w) b c', b=b, t=t)

                text_word_features = self.fusion_module_text(
                    tgt=text_word_features,
                    memory=src,
                    memory_key_padding_mask=mask,
                    pos=pos_l,
                    query_pos=None)

                # === Progressive 3-layer VLI again ===
                v_cur = src
                for i in range(3):
                    v_in = v_cur
                    v_out = self.vli_layers[i](
                        tgt=self.vli_norms[i](v_in),
                        memory=text_word_initial_features,
                        memory_key_padding_mask=text_word_masks,
                        pos=text_pos,
                        query_pos=None
                    )
                    v_cur = v_in + self.vli_res_scale * v_out
                src = v_cur

                src = rearrange(src, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
                mask = rearrange(mask, 'b (t h w) -> (b t) h w', t=t, h=h, w=w)
                pos_l = rearrange(pos_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)

        text_word_features = rearrange(text_word_features, 'l b c -> b l c')
        text_sentence_features = self.poolout_module(text_word_features)

        # Transformer
        query_embeds = self.query_embed.weight
        text_embed = repeat(text_sentence_features, 'b c -> b t q c', t=t, q=self.num_queries)
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples = \
            self.transformer(srcs, text_embed, masks, poses, query_embeds)

        out = {}
        outputs_classes, outputs_coords = [], []
        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_class = rearrange(outputs_class, 'l (b t) q k -> l b t q k', b=b, t=t)
        outputs_coord = rearrange(outputs_coord, 'l (b t) q n -> l b t q n', b=b, t=t)
        out['pred_logits'] = outputs_class[-1]
        out['pred_boxes'] = outputs_coord[-1]

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
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
            text_features = NestedTensor(text_features, text_masks)
        else:
            raise ValueError("Please make sure caption is list[str]")
        return text_features


class MLP(nn.Module):
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
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class FeatureResizer(nn.Module):
    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        return self.dropout(x)


def build(args):
    if args.binary:
        num_classes = 1
    else:
        if args.dataset_file == 'ytvos':
            num_classes = 65
        elif args.dataset_file == 'davis':
            num_classes = 78
        elif args.dataset_file in ['a2d', 'jhmdb']:
            num_classes = 1
        else:
            num_classes = 91
    device = torch.device(args.device)

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
        freeze_text_encoder=args.freeze_text_encoder
    )
    matcher = build_matcher(args)
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef
    }
    if args.masks:
        weight_dict['loss_mask'] = args.mask_loss_coef
        weight_dict['loss_dice'] = args.dice_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
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
    postprocessors = build_postprocessors(args, args.dataset_file)
    return model, criterion, postprocessors
