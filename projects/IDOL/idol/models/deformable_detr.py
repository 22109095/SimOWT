# ------------------------------------------------------------------------
# IDOL: In Defense of Online Models for Video Instance Segmentation
# Copyright (c) 2022 ByteDance. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from SeqFormer (https://github.com/wjf5203/SeqFormer)
# Copyright (c) 2021 ByteDance. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from ..util import box_ops
from ..util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher

from .segmentation_condInst import (CondInst_segm,
                           dice_loss, sigmoid_focal_loss)
                           
from .deformable_transformer import build_deforamble_transformer
import copy
from fvcore.nn import giou_loss, smooth_l1_loss

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def compute_box_iou(inputs, targets):
    """Compute pairwise iou between inputs, targets
    Both have the shape of [N, 4] and xyxy format
    """
    area1 = box_ops.box_area(inputs)
    area2 = box_ops.box_area(targets)

    lt = torch.max(inputs[:, None, :2], targets[:, :2])  # [N,M,2]
    rb = torch.min(inputs[:, None, 2:], targets[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)        # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    iou = torch.diag(iou)
    return iou

class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_frames, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, use_iou_branch=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_frames = num_frames
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        if use_iou_branch:
            self.iou_head = nn.Linear(hidden_dim, 1)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
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
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine  # idol default=True
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if use_iou_branch:
            self.iou_head.bias.data = torch.ones(1) * bias_value
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:  # [ATTN]
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            if use_iou_branch:
                self.iou_head = _get_clones(self.iou_head, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            if use_iou_branch:
                self.iou_head = nn.ModuleList([self.iou_head for _ in range(num_pred-1)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [num_frames x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]  # [ATTN] 这里写错了 没有+1
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
   
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        
        features, pos = self.backbone(samples)
        srcs = []
        masks = []
        poses = []
        for l, feat in enumerate(features[1:]):
            # src: [nf*N, _C, Hi, Wi],
            # mask: [nf*N, Hi, Wi],
            # pos: [nf*N, C, H_p, W_p]
            src, mask = feat.decompose() 
            src_proj_l = self.input_proj[l](src)    # src_proj_l: [nf*N, C, Hi, Wi]
            
            # src_proj_l -> [nf, N, C, Hi, Wi]
            n,c,h,w = src_proj_l.shape
            src_proj_l = src_proj_l.reshape(n//self.num_frames, self.num_frames, c, h, w).permute(1,0,2,3,4)
            
            # mask -> [nf, N, Hi, Wi]
            mask = mask.reshape(n//self.num_frames, self.num_frames, h, w).permute(1,0,2,3)
            
            # pos -> [nf, N, Hi, Wi]
            np, cp, hp, wp = pos[l+1].shape
            pos_l = pos[l+1].reshape(np//self.num_frames, self.num_frames, cp, hp, wp).permute(1,0,2,3,4)
            for n_f in range(self.num_frames):
                srcs.append(src_proj_l[n_f])
                masks.append(mask[n_f])
                poses.append(pos_l[n_f])
                assert mask is not None

        if self.num_feature_levels > (len(features) - 1):
            _len_srcs = len(features) - 1
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask    # [nf*N, H, W]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                
                # src -> [nf, N, C, H, W]
                n, c, h, w = src.shape
                src = src.reshape(n//self.num_frames, self.num_frames, c, h, w).permute(1,0,2,3,4)
                mask = mask.reshape(n//self.num_frames, self.num_frames, h, w).permute(1,0,2,3)
                np, cp, hp, wp = pos_l.shape
                pos_l = pos_l.reshape(np//self.num_frames, self.num_frames, cp, hp, wp).permute(1,0,2,3,4)

                for n_f in range(self.num_frames):
                    srcs.append(src[n_f])
                    masks.append(mask[n_f])
                    poses.append(pos_l[n_f])

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, poses, query_embeds)

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
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        print('pred_logits.shape = {}'.format(outputs_class[-1].shape))
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, mask_out_stride=4, num_frames=1):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses  # ['labels', 'boxes', 'masks','reid']
        self.focal_alpha = focal_alpha
        self.mask_out_stride = mask_out_stride
        self.num_frames = num_frames


    def loss_labels(self, outputs,  targets, ref_target, indices, num_boxes, files, log=True):
        """Classification lossm(NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        freeze = False
        if not freeze:
            assert 'pred_logits' in outputs

            src_logits = outputs['pred_logits']
            # print('src_logits.shape = {}'.format(src_logits.shape))  # [2, 300, 1]
            batch_size = len(targets)
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            src_logits_list = []
            target_classes_o_list = []
            for batch_idx in range(batch_size):
                valid_query = indices[batch_idx][0]
                gt_multi_idx = indices[batch_idx][1]
                if len(gt_multi_idx)==0:
                    continue
                bz_src_logits = src_logits[batch_idx]
                target_classes_o = targets[batch_idx]["labels"]
                target_classes[batch_idx,valid_query] =  target_classes_o[gt_multi_idx]

                src_logits_list.append(bz_src_logits[valid_query])
                target_classes_o_list.append(target_classes_o[gt_multi_idx])

            num_boxes = torch.cat(target_classes_o_list).shape[0] if len(target_classes_o_list) != 0 else 1

            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            # return
            target_classes_onehot = target_classes_onehot[:,:,:-1]  #
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) \
                      * src_logits.shape[1]
            losses = {'loss_ce': loss_ce}
            return losses
        else:
            return {'loss_ce': outputs['pred_boxes'].sum()*0}

    @torch.no_grad()
    def loss_cardinality(self, outputs,  targets, ref_target, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """

        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs,  targets, ref_target, indices, num_boxes ,ious1=[], files=[]):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        freeze = False
        if not freeze:
            assert 'pred_boxes' in outputs
            idx = self._get_src_permutation_idx(indices)
            src_boxes = outputs['pred_boxes']
            loss_boxiou = outputs['pred_boxes'].sum()*0
            if len(indices) == 1:
                if len(indices[0][1]) > 0:
                    src_boxes1 = outputs['pred_boxes'][0][indices[0][0]]
                    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
                    # box iou
                    if 'pred_boxious' in outputs:
                        with torch.no_grad():
                            ious = compute_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes1),
                                                   box_ops.box_cxcywh_to_xyxy(target_boxes))
                        tgt_iou_scores = ious
                        src_iou_scores = outputs['pred_boxious']  # [B, N, 1]
                        src_iou_scores = src_iou_scores[0][indices[0][0]]
                        src_iou_scores = src_iou_scores.flatten(0)
                        tgt_iou_scores = tgt_iou_scores.flatten(0)
                        loss_boxiou += F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean')
                else:
                    a = 1
            elif len(indices) == 2:
                if len(indices[0][1]) > 0:
                    src_boxes1 = outputs['pred_boxes'][0][indices[0][0]]
                    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip([targets[0]], [indices[0]])], dim=0)
                    # box iou
                    if 'pred_boxious' in outputs:
                        with torch.no_grad():
                            ious = compute_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes1),
                                                   box_ops.box_cxcywh_to_xyxy(target_boxes))
                        tgt_iou_scores = ious
                        src_iou_scores = outputs['pred_boxious']  # [B, N, 1]
                        src_iou_scores = src_iou_scores[0][indices[0][0]]
                        src_iou_scores = src_iou_scores.flatten(0)
                        tgt_iou_scores = tgt_iou_scores.flatten(0)
                        loss_boxiou += F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean')
                if len(indices[1][1]) > 0:
                    src_boxes1 = outputs['pred_boxes'][1][indices[1][0]]
                    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip([targets[1]], [indices[1]])], dim=0)
                    # box iou
                    if 'pred_boxious' in outputs:
                        with torch.no_grad():
                            ious = compute_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes1),
                                                   box_ops.box_cxcywh_to_xyxy(target_boxes))
                        tgt_iou_scores = ious
                        src_iou_scores = outputs['pred_boxious']  # [B, N, 1]
                        src_iou_scores = src_iou_scores[1][indices[1][0]]
                        src_iou_scores = src_iou_scores.flatten(0)
                        tgt_iou_scores = tgt_iou_scores.flatten(0)
                        loss_boxiou += F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean')
            else:
                a = 1

            batch_size = len(targets)
            pred_box_list = []
            tgt_box_list = []
            for batch_idx in range(batch_size):
                valid_query = indices[batch_idx][0]
                gt_multi_idx = indices[batch_idx][1]
                if len(gt_multi_idx)==0:
                    continue
                bz_src_boxes = src_boxes[batch_idx]
                bz_target_boxes = targets[batch_idx]["boxes"]
                pred_box_list.append(bz_src_boxes[valid_query])
                tgt_box_list.append(bz_target_boxes[gt_multi_idx])

            if len(pred_box_list) != 0:
                src_boxes = torch.cat(pred_box_list)
                target_boxes = torch.cat(tgt_box_list)
                num_boxes = src_boxes.shape[0]

                loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
                # a = 1.2
                a = 1.1
                # a = 1.05
                if ious1 != [] and len(ious1) == 1 :
                    try:
                        if files[0] == 'tao':
                            for idx, i in enumerate(loss_bbox[:len(ious1[0][1])]):
                                loss_bbox[idx] = a*loss_bbox[idx]*ious1[0][0][ious1[0][1][idx]]
                        # if files[1] == 'tao':
                        #     for idx, i in enumerate(loss_bbox[len(ious[0][1]):]):
                        #         loss_bbox[len(ious[0][1])+idx] = a*loss_bbox[len(ious[0][1])+idx]*ious[1][0][ious[1][1][idx]]
                        # print('true')
                    except:
                        print('error')
                        a = 1
                losses = {}
                losses['loss_bbox'] = loss_bbox.sum() / num_boxes
                loss_giou = giou_loss(box_ops.box_cxcywh_to_xyxy(src_boxes),box_ops.box_cxcywh_to_xyxy(target_boxes))

                if ious1 != [] and len(ious1) == 1 :
                    try:
                        if files[0] == 'tao':
                            for idx, i in enumerate(loss_giou[:len(ious1[0][1])]):
                                loss_giou[idx] = a*loss_giou[idx]*ious1[0][0][ious1[0][1][idx]]
                        # if files[1] == 'tao':
                        #     for idx, i in enumerate(loss_giou[len(ious[0][1]):]):
                        #         loss_giou[len(ious[0][1])+idx] = a*loss_giou[len(ious[0][1])+idx]*ious[1][0][ious[1][1][idx]]
                        # print('true')
                    except:
                        print('error')
                        a = 1
                losses['loss_giou'] = loss_giou.sum() / num_boxes
                if 'pred_boxious' in outputs:
                    losses['loss_boxiou'] = loss_boxiou
            else:
                losses = {'loss_bbox':outputs['pred_boxes'].sum()*0,
                'loss_giou':outputs['pred_boxes'].sum()*0}
                if 'pred_boxious' in outputs:
                    losses['loss_boxiou'] = outputs['pred_boxes'].sum()*0
            return losses
        else:
            return {'loss_bbox':outputs['pred_boxes'].sum()*0,
                'loss_giou':outputs['pred_boxes'].sum()*0}

        

    def loss_masks(self, outputs,  targets, ref_target, indices, num_boxes, files, only_det=False):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        freeze = False
        if not freeze:
            if not only_det:
                assert "pred_masks" in outputs
                if len(files) == 2:
                    if files[0] == 'coco' and files[1] == 'coco':
                        src_masks = outputs["pred_masks"]
                        coco = 1
                        coco_local = 0
                    elif files[0] == 'coco' and files[1] != 'coco':
                        src_masks = [outputs["pred_masks"][0]]
                        coco = 1
                        coco_local = 1
                    elif files[0] != 'coco' and files[1] == 'coco':
                        src_masks = [outputs["pred_masks"][1]]
                        coco = 1
                        coco_local = 2
                    else:
                        src_masks = outputs["pred_masks"]
                        coco = 0
                        coco_local = 0
                elif len(files) == 1:
                    if files[0] == 'coco':
                        src_masks = outputs["pred_masks"]
                        coco = 1
                        coco_local = 0
                    elif files[0] != 'coco':
                        src_masks = outputs["pred_masks"]
                        return {
                    "loss_mask": (src_masks[0]).sum()*0,
                    "loss_dice": (src_masks[0]).sum()*0,
                    }
                if type(src_masks) == list:
                    src_masks = torch.cat(src_masks, dim=1)[0]
                key_frame_masks = [t["masks"] for t in targets]
                ref_frame_masks = [t["masks"] for t in ref_target]
                #during coco pretraining, the sizes of two input frames are different, so we pad them together ant get key frame gt mask for loss calculation
                target_masks, valid = nested_tensor_from_tensor_list(key_frame_masks+ref_frame_masks,
                                                                     size_divisibility=32,
                                                                     split=False).decompose()
                if coco_local == 1:
                    target_masks = target_masks[:1]
                    targets = [targets[0]]
                    indices = [indices[0]]
                elif coco_local == 2:
                    target_masks = target_masks[1:2]
                    targets = [targets[1]]
                    indices = [indices[1]]
                else:
                    target_masks = target_masks[:len(key_frame_masks)]

                target_masks = target_masks.to(src_masks)
                # downsample ground truth masks with ratio mask_out_stride
                start = int(self.mask_out_stride // 2)
                im_h, im_w = target_masks.shape[-2:]

                target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]

                assert target_masks.size(2) * self.mask_out_stride == im_h
                assert target_masks.size(3) * self.mask_out_stride == im_w

                batch_size = len(targets)
                tgt_mask_list = []
                for batch_idx in range(batch_size):
                    valid_num = targets[batch_idx]["masks"].shape[0]
                    gt_multi_idx = indices[batch_idx][1]
                    if len(gt_multi_idx)==0:
                        continue
                    batch_masks = target_masks[batch_idx][:valid_num][gt_multi_idx].unsqueeze(1)
                    tgt_mask_list.append(batch_masks)


                if len(tgt_mask_list) != 0:
                    target_masks = torch.cat(tgt_mask_list)
                    num_boxes = src_masks.shape[0]
                    assert src_masks.shape == target_masks.shape

                    src_masks = src_masks.flatten(1)
                    target_masks = target_masks.flatten(1)
                    losses = {
                        "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes)*coco,
                        "loss_dice": dice_loss(src_masks, target_masks, num_boxes)*coco,
                    }
                else:
                    losses = {
                    "loss_mask": (src_masks).sum()*coco,
                    "loss_dice": (src_masks).sum()*coco,
                    }
                return losses
            else:
                assert "pred_masks" in outputs

                if files[0] == 'coco' and files[1] == 'coco':
                    src_masks = outputs["pred_masks"]
                    coco = 1
                    coco_local = 0
                elif files[0] == 'coco' and files[1] != 'coco':
                    src_masks = [outputs["pred_masks"][0]]
                    coco = 1
                    coco_local = 1
                elif files[0] != 'coco' and files[1] == 'coco':
                    src_masks = [outputs["pred_masks"][1]]
                    coco = 1
                    coco_local = 2
                else:
                    src_masks = outputs["pred_masks"]
                    coco = 0
                    coco_local = 0
                if type(src_masks) == list:
                    src_masks = torch.cat(src_masks, dim=1)[0]
                key_frame_masks = [t["masks"] for t in targets]
                if key_frame_masks[0].shape[0]==0:
                    return {
                        "loss_mask": (src_masks * 0).sum() * coco,
                        "loss_dice": (src_masks * 0).sum() * coco,
                    }

                target_masks, valid = nested_tensor_from_tensor_list(key_frame_masks,
                                                                     size_divisibility=32,
                                                                     split=False, shape=src_masks.shape, mask_loss=True).decompose()

                if coco_local == 1:
                    target_masks = target_masks[:1]
                    targets = [targets[0]]
                    indices = [indices[0]]
                elif coco_local == 2:
                    target_masks = target_masks[1:2]
                    targets = [targets[1]]
                    indices = [indices[1]]
                else:
                    target_masks = target_masks[:len(key_frame_masks)]

                target_masks = target_masks.to(src_masks)
                # downsample ground truth masks with ratio mask_out_stride
                start = int(self.mask_out_stride // 2)
                im_h, im_w = target_masks.shape[-2:]

                target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]

                assert target_masks.size(2) * self.mask_out_stride == im_h
                assert target_masks.size(3) * self.mask_out_stride == im_w

                batch_size = len(targets)
                tgt_mask_list = []
                for batch_idx in range(batch_size):
                    valid_num = targets[batch_idx]["masks"].shape[0]
                    gt_multi_idx = indices[batch_idx][1]
                    if len(gt_multi_idx) == 0:
                        continue
                    batch_masks = target_masks[batch_idx][:valid_num][gt_multi_idx].unsqueeze(1)
                    tgt_mask_list.append(batch_masks)

                if len(tgt_mask_list) != 0:
                    target_masks = torch.cat(tgt_mask_list)
                    num_boxes = src_masks.shape[0]
                    assert src_masks.shape == target_masks.shape
                    src_masks = src_masks.flatten(1)
                    target_masks = target_masks.flatten(1)

                    losses = {
                        "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes) * coco,
                        "loss_dice": dice_loss(src_masks, target_masks, num_boxes) * coco,
                    }
                else:
                    losses = {
                        "loss_mask": (src_masks * 0).sum() * coco,
                        "loss_dice": (src_masks * 0).sum() * coco,
                    }
                return losses
        else:
            return {
                        "loss_mask": (outputs['pred_boxes']).sum()*0,
                        "loss_dice": (outputs['pred_boxes']).sum()*0,
                    }


    def loss_reid(self, outputs,  targets, ref_target, indices, num_boxes, files, ious, tags, matched_reid):
        qd_items = outputs['pred_qd']
        contras_loss = 0
        aux_loss = 0
        if len(qd_items) == 0:
            losses = {'loss_reid': outputs['pred_logits'].sum()*0,
                   'loss_reid_aux':  outputs['pred_logits'].sum()*0 }
            return losses
        try:
            longscore = torch.cat((targets[0]['longscore'], targets[1]['longscore']))
        except:
            longscore = targets[0]['longscore']
        try:
            assert len(longscore)==len(qd_items)
        except:
            a = 1
        longscore1 = []
        a = 3
        if len(matched_reid[0])==1 and matched_reid[0][0]==False:
            matched_reid1 = matched_reid[1]
            a = 1
        if len(matched_reid[1])==1 and matched_reid[1][0]==False:
            matched_reid1 = matched_reid[0]
            a = 2
        if a == 3:
            matched_reid1 = matched_reid[0]+matched_reid[1]
        try:
            for idx, item in enumerate(matched_reid1):
                if item == True:
                    longscore1.append(float(math.sqrt(longscore[idx])))
        except:
            a = 1
        if files[0]=='coco':
            longscore1[:len(matched_reid[0])] = [1]*len(matched_reid[0])
        try:
            if files[1]=='coco':
                longscore1[:len(matched_reid[1])] = [1]*len(matched_reid[1])
        except:
            a = 1
        for idx, qd_item in enumerate(qd_items):
            pred = qd_item['contrast'].permute(1,0)
            label = qd_item['label'].unsqueeze(0)
            # contrastive loss
            pos_inds = (label == 1)
            neg_inds = (label == 0)
            pred_pos = pred * pos_inds.float()
            pred_neg = pred * neg_inds.float()
            # use -inf to mask out unwanted elements.
            pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
            pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

            _pos_expand = torch.repeat_interleave(pred_pos, pred.shape[1], dim=1)
            _neg_expand = pred_neg.repeat(1, pred.shape[1])
            # [bz,N], N is all pos and negative samples on reference frame, label indicate it's pos or negative
            x = torch.nn.functional.pad((_neg_expand - _pos_expand), (0, 1), "constant", 0) 
            # contras_loss += torch.logsumexp(x, dim=1)
            contras_loss += torch.logsumexp(x, dim=1)*longscore1[idx]

            aux_pred = qd_item['aux_consin'].permute(1,0)
            aux_label = qd_item['aux_label'].unsqueeze(0)

            # aux_loss += (torch.abs(aux_pred - aux_label)**2).mean()
            aux_loss += (torch.abs(aux_pred - aux_label)**2).mean()*longscore1[idx]

        losses = {'loss_reid': contras_loss.sum()/len(qd_items),
                   'loss_reid_aux':  aux_loss/len(qd_items) }

        return losses

    

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

    def get_loss(self, loss, outputs, targets, ref_target, indices, num_boxes, ious, files, tags, matched_reid, only_det=False, **kwargs):
        if not only_det:
            loss_map = {
                'labels': self.loss_labels,
                'cardinality': self.loss_cardinality,
                'boxes': self.loss_boxes,
                'masks': self.loss_masks,
                'reid': self.loss_reid
            }
            assert loss in loss_map, f'do you really want to compute {loss} loss?'
            if loss == 'boxes':
                return loss_map[loss](outputs,  targets, ref_target, indices, num_boxes, ious, files, **kwargs)
            elif loss == 'masks':
                return loss_map[loss](outputs, targets, ref_target, indices, num_boxes, files, **kwargs)
            elif loss == 'reid':
                return loss_map[loss](outputs, targets, ref_target, indices, num_boxes, files, ious, tags, matched_reid, **kwargs)
            else:
                return loss_map[loss](outputs, targets, ref_target, indices, num_boxes, files, **kwargs)
        else:
            loss_map = {
                'labels': self.loss_labels,
                'cardinality': self.loss_cardinality,
                'boxes': self.loss_boxes,
                'masks': self.loss_masks,
            }
            assert loss in loss_map, f'do you really want to compute {loss} loss?'
            if loss == 'boxes':
                return loss_map[loss](outputs, targets, ref_target, indices, num_boxes, ious, files, **kwargs)
            elif loss == 'masks':
                return loss_map[loss](outputs, targets, ref_target, indices, num_boxes, files, only_det=only_det, **kwargs)
            else:
                return loss_map[loss](outputs, targets, ref_target, indices, num_boxes, **kwargs)
    def forward(self, outputs, targets, ref_target, indices_list, ious_list, files, file_names, tags, matched_reid, only_det=False):  # [ATTN]
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        if not only_det:
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

            # Compute all the requested losses
            losses = {}
            for loss in self.losses:
                kwargs = {}
                losses.update(self.get_loss(loss, outputs, targets, ref_target, indices_list[-1], num_boxes, ious_list[-1], files, tags, matched_reid, **kwargs))

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    # indices = self.matcher(aux_outputs, targets)
                    indices = indices_list[i]
                    ious= ious_list[i]
                    for loss in self.losses:
                        # if loss == 'reid' or 'masks':
                        if loss == 'reid':
                            continue
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs['log'] = False
                        l_dict = self.get_loss(loss, aux_outputs, targets, ref_target, indices, num_boxes, ious, files, tags, matched_reid, **kwargs)
                        # l_dict = self.get_loss(loss, aux_outputs, targets, ref_target, indices, num_boxes, [], files, tags, matched_reid, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
            if 'enc_outputs' in outputs:
                enc_outputs = outputs['enc_outputs']
                bin_targets = copy.deepcopy(targets)
                for bt in bin_targets:
                    bt['labels'] = torch.zeros_like(bt['labels'])
                # We do not use OTA for the first stage
                indices, matched_ids, ious = self.matcher.forward(enc_outputs, bin_targets, file_names)
                # indices = self.matcher.forward_h(enc_outputs, bin_targets, [])
                for loss in self.losses:
                    if loss == 'masks' or loss == 'reid':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, enc_outputs, bin_targets, ref_target, indices, num_boxes, [], files, **kwargs)
                    l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                    losses.update(l_dict)
            return losses
        else:
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

            # Compute all the requested losses
            losses = {}
            for loss in self.losses:
                kwargs = {}
                losses.update(
                    self.get_loss(loss, outputs, targets, ref_target, indices_list[-1], num_boxes, ious_list[-1], files,
                                  tags, only_det=only_det, **kwargs))

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    # indices = self.matcher(aux_outputs, targets)
                    indices = indices_list[i]
                    ious = ious_list[i]
                    for loss in self.losses:
                        # if loss == 'reid' or 'masks':
                        if loss == 'reid':
                            continue
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs['log'] = False
                        # l_dict = self.get_loss(loss, aux_outputs, targets, ref_target, indices, num_boxes, ious, **kwargs)
                        l_dict = self.get_loss(loss, aux_outputs, targets, ref_target, indices, num_boxes, [], files,
                                               tags, only_det=only_det, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
            if 'enc_outputs' in outputs:
                enc_outputs = outputs['enc_outputs']
                bin_targets = copy.deepcopy(targets)
                for bt in bin_targets:
                    bt['labels'] = torch.zeros_like(bt['labels'])
                # if self.ota:
                #     indices, matched_ids = self.matcher.forward_ota(enc_outputs, bin_targets)
                # else:
                # We do not use OTA for the first stage
                indices, matched_ids, ious = self.matcher.forward(enc_outputs, bin_targets, file_names)
                # indices = self.matcher.forward_h(enc_outputs, bin_targets, [])
                for loss in self.losses:
                    if loss == 'masks' or loss == 'reid':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, enc_outputs, bin_targets, ref_target, indices, num_boxes, [], files,
                                           **kwargs)
                    l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                    losses.update(l_dict)
            return losses



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


