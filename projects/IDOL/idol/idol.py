from detectron2.structures import Boxes, BoxMode, Instances
import math
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import cv2
from typing import Dict, List
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from fvcore.nn import giou_loss, smooth_l1_loss
from .models.backbone import Joiner
from .models.deformable_detr import DeformableDETR, SetCriterion
from .models.matcher import HungarianMatcher
from .models.position_encoding import PositionEmbeddingSine
from .models.deformable_transformer import DeformableTransformer
from .models.segmentation_condInst import CondInst_segm, segmentation_postprocess
from .models.tracker import IDOL_Tracker
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import NestedTensor
from .data.coco import convert_coco_poly_to_mask
import torchvision.ops as ops
import json
import time
import matplotlib
from matplotlib.font_manager import *
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
from PIL import Image
from numpy import average, dot, linalg
__all__ = ["IDOL"]
def _region_foreground_background_masking(feat, mask):
    T, C, height, width = feat.size()
    mask = F.interpolate(mask, size=(height, width), mode='nearest')
    temporal_feat_fg_vertor = []
    temporal_feat_bg_vertor = []
    for t in range(T):

        one_feat = feat[t:t + 1, ...]
        one_mask = mask[t:t + 1, ...]

        fore_pixel_num = (one_mask == 1).sum()

        back_pixel_num = (one_mask == 0).sum()

        feat_fg = one_feat * one_mask  # [1, c, h, w]

        feat_bg = one_feat * (1 - one_mask)  # [1, c, h, w]

        feat_fg_vertor = (feat_fg.flatten(2).sum(-1)) / (fore_pixel_num + 1e-8)  # [1,96]

        feat_bg_vector = (feat_bg.flatten(2).sum(-1)) / (back_pixel_num + 1e-8)
        temporal_feat_fg_vertor.append(feat_fg_vertor)
        temporal_feat_bg_vertor.append(feat_bg_vector)
    temporal_feat_fg_vertor = torch.cat(temporal_feat_fg_vertor, dim=0)
    temporal_feat_bg_vertor = torch.cat(temporal_feat_bg_vertor, dim=0)
    return temporal_feat_fg_vertor, temporal_feat_bg_vertor

def classify_hist_with_split(image1, image2, size=(256, 256)):
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data

def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])

    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

def img_similarity(img1, img2):
    w1, h1 = img1.shape
    w2, h2 = img2.shape
    img1 = cv2.resize(img1, (h1, w1))
    img2 = cv2.resize(img2, (h2, w2))

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)

    good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
    similary = float(len(good)) / len(matches)
    if similary > 0.1:
        print(similary)
    else:
        print(similary)
    return similary


def get_thum(image, size=(64, 64), greyscale=False):
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        image = image.convert('L')
    return image

def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)
    return res



class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8');
        return json.JSONEncoder.default(self, obj)

def mask_iou(mask1, mask2):
    mask1 = mask1.char()
    mask2 = mask2.char()

    intersection = (mask1[:,:,:] * mask2[:,:,:]).sum(-1).sum(-1)
    union = (mask1[:,:,:] + mask2[:,:,:] - mask1[:,:,:] * mask2[:,:,:]).sum(-1).sum(-1)

    return (intersection+1e-6) / (union+1e-6)

def compute_size(det1 ,det2):
    x1, y1, x2, y2 = det1[:4]
    x3, y3, x4, y4 = det2[:4]

    if x2<=x3 or x4<=x1 or y2 <= y3 or y4 <= y1:
        return 0
    else:
        lens = min(x2, x4) - max(x1, x3)
        wide = min(y2, y4) - max(y1, y3)
        if lens * wide<0:
            print(1)
        return lens * wide

def iou_score(x1, y1, x2, y2, a1, b1, a2, b2):
    ax = max(x1, a1)
    ay = max(y1, b1)
    bx = min(x2, a2)
    by = min(y2, b2)

    area_N = (x2 - x1) * (y2 - y1)
    area_M = (a2 - a1) * (b2 - b1)

    w = bx - ax
    h = by - ay
    if w <= 0 or h <= 0:
        return 0
    area_X = w * h
    return area_X / (area_N + area_M - area_X)

def mask_nms1(seg_masks, scores, category_ids, nms_thr=0.5):
    n_samples = len(scores)
    if n_samples == 0:
        return []
    keep = [True for i in range(n_samples)]
    seg_masks = seg_masks.sigmoid() > 0.5

    for i in range(n_samples - 1):
        if not keep[i]:
            continue
        mask_i = seg_masks[i]
        for j in range(i + 1, n_samples, 1):
            if not keep[j]:
                continue
            mask_j = seg_masks[j]

            iou = mask_iou(mask_i, mask_j)[0]
            if iou > nms_thr:
                keep[j] = False
    return keep

class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = [backbone_shape[f].channels for f in backbone_shape.keys()]
       
    def forward(self, tensor_list):
        xs = self.backbone(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


@META_ARCH_REGISTRY.register()
class IDOL(nn.Module):
    """
    Implement IDOL
    """

    def __init__(self, cfg):
        super().__init__()

        self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.clip_stride = cfg.MODEL.IDOL.CLIP_STRIDE

        ### inference setting
        self.merge_on_cpu = cfg.MODEL.IDOL.MERGE_ON_CPU
        self.is_multi_cls = cfg.MODEL.IDOL.MULTI_CLS_ON
        self.apply_cls_thres = cfg.MODEL.IDOL.APPLY_CLS_THRES
        self.temporal_score_type = cfg.MODEL.IDOL.TEMPORAL_SCORE_TYPE
        self.inference_select_thres = cfg.MODEL.IDOL.INFERENCE_SELECT_THRES
        self.inference_fw = cfg.MODEL.IDOL.INFERENCE_FW
        self.inference_tw = cfg.MODEL.IDOL.INFERENCE_TW
        self.memory_len = cfg.MODEL.IDOL.MEMORY_LEN
        self.batch_infer_len = cfg.MODEL.IDOL.BATCH_INFER_LEN
        self.class_agnostic = cfg.MODEL.CLASS_AGNOSTIC


        self.is_coco = cfg.DATASETS.TEST[0].startswith("coco")
        self.num_classes = cfg.MODEL.IDOL.NUM_CLASSES  # 1
        self.mask_stride = cfg.MODEL.IDOL.MASK_STRIDE
        self.match_stride = cfg.MODEL.IDOL.MATCH_STRIDE
        self.mask_on = cfg.MODEL.MASK_ON

        self.coco_pretrain = cfg.INPUT.COCO_PRETRAIN
        hidden_dim = cfg.MODEL.IDOL.HIDDEN_DIM
        num_queries = cfg.MODEL.IDOL.NUM_OBJECT_QUERIES

        # Transformer parameters:
        nheads = cfg.MODEL.IDOL.NHEADS
        dropout = cfg.MODEL.IDOL.DROPOUT
        dim_feedforward = cfg.MODEL.IDOL.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.IDOL.ENC_LAYERS
        dec_layers = cfg.MODEL.IDOL.DEC_LAYERS
        enc_n_points = cfg.MODEL.IDOL.ENC_N_POINTS
        dec_n_points = cfg.MODEL.IDOL.DEC_N_POINTS
        num_feature_levels = cfg.MODEL.IDOL.NUM_FEATURE_LEVELS

        # Loss parameters:
        mask_weight = cfg.MODEL.IDOL.MASK_WEIGHT
        dice_weight = cfg.MODEL.IDOL.DICE_WEIGHT
        giou_weight = cfg.MODEL.IDOL.GIOU_WEIGHT
        l1_weight = cfg.MODEL.IDOL.L1_WEIGHT
        class_weight = cfg.MODEL.IDOL.CLASS_WEIGHT
        reid_weight = cfg.MODEL.IDOL.REID_WEIGHT
        deep_supervision = cfg.MODEL.IDOL.DEEP_SUPERVISION

        focal_alpha = cfg.MODEL.IDOL.FOCAL_ALPHA

        set_cost_class = cfg.MODEL.IDOL.SET_COST_CLASS
        set_cost_bbox = cfg.MODEL.IDOL.SET_COST_BOX
        set_cost_giou = cfg.MODEL.IDOL.SET_COST_GIOU
        use_iou_branch = True
        self.only_det = False
        self.reid_depth = True
        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels[1:]
        backbone.strides = d2_backbone.feature_strides[1:]

        self.two_stage = False
        self.two_stage_num_proposals = 300
        transformer = DeformableTransformer(
        d_model= hidden_dim,
        nhead=nheads,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_frames=self.num_frames,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points,
        enc_n_points=enc_n_points,
        two_stage=self.two_stage,
        two_stage_num_proposals=self.two_stage_num_proposals)


        model = DeformableDETR(
        backbone,
        transformer,
        num_classes=self.num_classes,
        num_frames=self.num_frames,
        num_queries=num_queries,
        num_feature_levels=num_feature_levels,
        aux_loss=deep_supervision,
        with_box_refine=True,
        two_stage=self.two_stage,
        use_iou_branch=use_iou_branch)

        self.detr = CondInst_segm(model, freeze_detr=False, rel_coord=True, use_iou_branch=use_iou_branch, reid_depth=self.reid_depth, only_det=self.only_det)
        self.detr.to(self.device)


        # building criterion
        matcher = HungarianMatcher(multi_frame=True,
                            cost_class=set_cost_class,
                            cost_bbox=set_cost_bbox,
                            cost_giou=set_cost_giou)
        boxiou_weight = 2.0
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou":giou_weight, 'loss_boxiou':boxiou_weight}
        if not self.only_det:
            weight_dict["loss_reid"] = reid_weight
            weight_dict["loss_reid_aux"] = reid_weight*1.5
        weight_dict["loss_mask"] = mask_weight
        weight_dict["loss_dice"] = dice_weight

        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        if not self.only_det:
            losses = ['labels', 'boxes', 'masks','reid']
        else:
            losses = ['labels', 'boxes', 'masks']


        self.criterion = SetCriterion(self.num_classes, matcher, weight_dict, losses,
                             mask_out_stride=self.mask_stride,
                             focal_alpha=focal_alpha,
                             num_frames = self.num_frames)
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        self.merge_device = "cpu" if self.merge_on_cpu else self.device
        self.img_count = 0
        path_tao_val = './datasets/tao/annotations/val_split/all.json'
        with open(path_tao_val, 'r') as f2:
            tao_all = json.load(f2)
        self.json = tao_all['videos']
        self.json_gen = []
        self.empty = 0
        self.toomuch = 0
        self.num_pse = 0
        self.num_pse2 = 0
        self.num_pse3 = 0
        self.num_pse4 = 0
        self.num_pse_coco = 0
        self.num_pse_noncoco = 0
        self.anno = []
        self.img = []
        self.tracklets1 = []
        self.datasrc_pre = ''
        self.datasrc = ''
        self.video_pre = ''
        self.video = ''
        self.frame_id = 0


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        if self.training:
            file_names = []
            tags = []
            for item in batched_inputs:
                file_names.append([item['file_name'], item['height'], item['width']])
                tag = (item['instances'][0]._fields['gt_classes']/10000).int().numpy().tolist()
                item['instances'][0]._fields['gt_classes'] = item['instances'][0]._fields['gt_classes']%10000
                item['instances'][1]._fields['gt_classes'] = item['instances'][1]._fields['gt_classes']%10000

                item['instances'][0]._fields['longscore'] = item['instances'][0]._fields['gt_classes']/1000
                item['instances'][1]._fields['longscore'] = item['instances'][1]._fields['gt_classes']/1000
                item['instances'][0]._fields['gt_classes'] = item['instances'][0]._fields['gt_classes'] - item['instances'][0]._fields['gt_classes']
                item['instances'][1]._fields['gt_classes'] = item['instances'][1]._fields['gt_classes'] - item['instances'][1]._fields['gt_classes']
                tags.append(tag)
            images = self.preprocess_image(batched_inputs)
            gt_instances = []
            for video in batched_inputs:
                for frame in video["instances"]:
                    gt_instances.append(frame.to(self.device))
            det_targets,ref_targets, tags1 = self.prepare_targets(gt_instances, tags)

            files = []
            if 'train2017' in batched_inputs[0]['file_name']:
                files.append('coco')
            else:
                files.append('tao')
            # if 'train2017' in batched_inputs[1]['file_name']:
            #     files.append('coco')
            # else:
            #     files.append('tao')

            output, loss_dict = self.detr(images, det_targets,ref_targets, file_names, files, self.criterion, tags1, train=True, only_det=False)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        elif self.coco_pretrain:  #evluate during coco pretrain
            # local
            function_choice = 2
            batched_inputs = batched_inputs[0]

            images = self.preprocess_coco_image(batched_inputs)
            output, features = self.detr.inference_forward(images, size_divisib=32, only_det=self.only_det) #
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"] if self.mask_on else None
            if not self.only_det:
                reid_pred = output['pred_inst_embed']
            else:
                reid_pred = None
            if self.detr.use_iou_branch:
                iou_pred = output["pred_boxious"]
            else:
                iou_pred = None
            results, box_i, mask_pred_i, reid_pred_i = self.coco_inference(box_cls, box_pred, mask_pred, images.image_sizes, reid_pred, iou_pred=iou_pred)
            processed_results = []
            try:
                for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = segmentation_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
            except:
                for results_per_image, input_per_image, image_size in zip(results, [batched_inputs], images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = segmentation_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
            if function_choice == 1:
                self.original_pseudo(processed_results, box_cls, box_pred, mask_pred,
                                reid_pred, iou_pred, box_i, [], [], batched_inputs, [], image_size)
            elif function_choice == 2:
                self.track_eval(processed_results, box_cls, box_pred, mask_pred,
                           reid_pred, iou_pred, box_i, mask_pred_i, reid_pred_i, [], [], batched_inputs, [], image_size)
            elif function_choice == 3:
                self.pseudo_boxes_merging_box_level(processed_results, box_cls, box_pred, mask_pred,
                           reid_pred, iou_pred, box_i, [], [], batched_inputs, [], image_size)
            if function_choice > 0:
                return []
            return processed_results
        else:
            images = self.preprocess_image(batched_inputs)
            video_len = 1
            clip_length = self.batch_infer_len
            if video_len > clip_length:
                num_clips = math.ceil(video_len/clip_length)
                logits_list, boxes_list, embed_list, points_list, masks_list = [], [], [], [], []
                for c in range(num_clips):
                    start_idx = c*clip_length
                    end_idx = (c+1)*clip_length
                    clip_inputs = [{'image':batched_inputs[0]['image'][start_idx:end_idx]}]
                    clip_images = self.preprocess_image(clip_inputs)
                    clip_output = self.detr.inference_forward(clip_images)
                    logits_list.append(clip_output['pred_logits'])
                    boxes_list.append(clip_output['pred_boxes'])
                    embed_list.append(clip_output['pred_inst_embed'])
                    masks_list.append(clip_output['pred_masks'].to(self.merge_device))
                output = {
                    'pred_logits':torch.cat(logits_list,dim=0),
                    'pred_boxes':torch.cat(boxes_list,dim=0),
                    'pred_inst_embed':torch.cat(embed_list,dim=0),
                    'pred_masks':torch.cat(masks_list,dim=0),
                }    
            else:
                images = self.preprocess_image(batched_inputs)
                output = self.detr.inference_forward(images)
            if self.img_count == 0:
                self.idol_tracker = IDOL_Tracker(
                    init_score_thr=0.2,
                    obj_score_thr=0.1,
                    nms_thr_pre=0.5,
                    nms_thr_post=0.05,
                    addnew_score_thr=0.2,
                    memo_tracklet_frames=10,
                    memo_momentum=0.8,
                    long_match=self.inference_tw,
                    frame_weight=(self.inference_tw | self.inference_fw),
                    temporal_weight=self.inference_tw,
                    memory_len=self.memory_len
                )
                self.img_count += 1
            height = batched_inputs[0]['height']
            width = batched_inputs[0]['width']
            video_output = self.inference(output, self.idol_tracker, (height, width), images.image_sizes[0])  # (height, width) is resized size,images. image_sizes[0] is original size

            return video_output

    def original_pseudo(self,processed_results, box_cls, box_pred, mask_pred,
                        reid_pred, iou_pred, box_i, datasets_ori, idx_ori, batched_inputs, item, image_size):
        width_select = batched_inputs[0]['width']
        height_select = batched_inputs[0]['height']


        logits = box_cls[0].sigmoid()
        output_boxes = box_pred[0]
        output_mask = mask_pred[0]
        scores = logits.cpu().detach()
        max_score, _ = torch.max(logits, 1)
        indices = torch.nonzero(max_score > self.inference_select_thres, as_tuple=False).squeeze(1)
        if len(indices) == 0:
            topkv, indices_top1 = torch.topk(scores.max(1)[0], k=1)
            indices_top1 = indices_top1[torch.argmax(topkv)]
            indices = [indices_top1.tolist()]
        else:
            nms_scores, idxs = torch.max(logits[indices], 1)
            boxes_before_nms = box_cxcywh_to_xyxy(output_boxes[indices])
            keep_indices = ops.batched_nms(boxes_before_nms, nms_scores, idxs, 0.9)
            indices = indices[keep_indices]
        box_score = torch.max(logits[indices], 1)[0]
        det_bboxes = torch.cat([output_boxes[indices], box_score.unsqueeze(1)], dim=1)  # [ATTN] 这里将score 拼到了bbox的最后一维
        det_labels = torch.argmax(logits[indices], dim=1)
        det_masks = output_mask[indices]
        valids = mask_nms1(det_masks, det_bboxes[:, -1], None, 0.3)
        det_bboxes = det_bboxes[valids, :]
        init_inds = det_bboxes[:, 4] > 0.30
        det_bboxes = det_bboxes[init_inds, :]
        for idx_pse in range(len(det_bboxes)):
            bbox = det_bboxes[idx_pse]
            i = bbox
            c_x = int(i[0] * width_select)
            c_y = int(i[1] * height_select)
            w = int(i[2] * width_select)
            h = int(i[3] * height_select)
            x1 = max(0, int(c_x - 0.5 * w))
            y1 = max(0, int(c_y - 0.5 * h))
            x2 = max(0, int(c_x + 0.5 * w))
            y2 = max(0, int(c_y + 0.5 * h))
            if w<15 or h<15 or (h*w)>(height_select*width_select*0.7):
                continue
            tl_x = x1
            tl_y = y1
            self.anno.append({'file_name': batched_inputs[0]['file_name'].replace(
                'datasets/UVO/uvo_videos_dense_frames/datasets/UVO/uvo_videos_dense_frames/',
                'datasets/UVO/uvo_videos_dense_frames/'),
                              'image_id': batched_inputs[0]['image_id'],
                              'segmentation': [[tl_x, tl_y, tl_x + w, tl_y, tl_x + w, tl_y + h, tl_x, tl_y + h]],
                              'bbox': [tl_x, tl_y, w, h],
                              'id': self.num_pse + int(int(box_i.device.index) + 1) * 10000000000,
                              'category_id': 1,
                              'gen': 1,
                              'longscore': float(bbox[4]),
                              'iou_score': 0,
                              'iscrowd': 0,
                              'isgt_wby': 0,
                              'epoch0_gt': 1
                              })
            self.num_pse += 1

        if len(self.anno) > 0:
            dst_root = ''.format(str(30))
            if not os.path.exists(dst_root):
                os.makedirs(dst_root)
            try:
                train_coco_path = dst_root + str(batched_inputs[0]['image_id']).zfill(10) + '.json'
                with open(train_coco_path, 'w') as f3:
                    json.dump(self.anno, f3)
            except:
                file = open(train_coco_path, 'w', encoding='utf-8');
                file.write(json.dumps(self.anno, cls=MyEncoder, indent=4)) 
        else:
            self.empty += 1
        self.anno = []
    def track_eval(self, processed_results, box_cls, box_pred, mask_pred,
                        reid_pred, iou_pred, box_i, mask_pred_i, reid_pred_i, datasets_ori, idx_ori, batched_inputs, item, image_size):

        file_name_for_video = batched_inputs[0]['file_name']
        file_name_list = file_name_for_video.split('/')
        self.datasrc = file_name_list[4]
        self.video = file_name_list[5]
        init_score_thr = 0.2
        obj_score_thr = 0.1
        nms_thr_pre = 0.5
        # init_score_thr = 0.7
        # obj_score_thr = 0.5
        # nms_thr_pre = 0.3
        if self.frame_id == 0:
            self.datasrc_pre = file_name_list[4]
            self.video_pre = file_name_list[5]
            self.idol_tracker = IDOL_Tracker(
                init_score_thr=init_score_thr,
                obj_score_thr=obj_score_thr,
                nms_thr_pre=nms_thr_pre,
                nms_thr_post=0.05,
                addnew_score_thr=0.2,
                memo_tracklet_frames=10,
                memo_momentum=0.8,
                long_match=True,
                frame_weight=True,
                temporal_weight=True,
                memory_len=3
            )
        if self.video_pre != self.video:
            self.frame_id = 0
            self.idol_tracker = IDOL_Tracker(
                init_score_thr=init_score_thr,
                obj_score_thr=obj_score_thr,
                nms_thr_pre=nms_thr_pre,
                nms_thr_post=0.05,
                addnew_score_thr=0.2,
                memo_tracklet_frames=10,
                memo_momentum=0.8,
                long_match=True,
                frame_weight=True,
                temporal_weight=True,
                memory_len=3
            )
            self.video_pre = self.video
            self.datasrc_pre = self.datasrc
        file_name_1 = batched_inputs[0]['file_name']
        video_name_list = file_name_1.split('/')
        # tao
        video_name_1 = video_name_list[3] + '/' + video_name_list[4] + '/' + video_name_list[5]
        for video in self.json:
            if video_name_1 == video['name']:
                video_id = video['id']
                break
        self.tracklets1 = []
        device = box_cls.device
        det_bboxes = box_i.to(device)
        scores = processed_results[0]['instances']._fields['scores'].unsqueeze(1).to(device)
        indices = torch.arange(len(det_bboxes)).to(device)
        det_labels = torch.zeros(len(det_bboxes)).to(device)
        det_masks = mask_pred_i.to(device)
        track_feats = reid_pred_i.to(device)

        if len(indices) > 0:
            max_score, _ = torch.max(scores.sigmoid(), 1)
            indices = torch.nonzero(max_score > self.inference_select_thres, as_tuple=False).squeeze(1)
            if len(indices) == 0:  # 如果没有超过阈值的则选择最大的
                topkv, indices_top1 = torch.topk(scores.max(1)[0], k=1)
                indices_top1 = indices_top1[torch.argmax(topkv)]
                indices = [indices_top1.tolist()]
            else:
                nms_scores, idxs = torch.max(scores.sigmoid()[indices], 1)
                boxes_before_nms = box_cxcywh_to_xyxy(det_bboxes[:, :4][indices])
                keep_indices = ops.batched_nms(boxes_before_nms, nms_scores, idxs, 0.9)
                indices = indices[keep_indices]
            box_score = torch.max(scores.sigmoid()[indices], 1)[0]
            det_bboxes = torch.cat([det_bboxes[indices], box_score.unsqueeze(1)],
                                   dim=1)
            det_labels = torch.argmax(scores.sigmoid()[indices], dim=1)
            track_feats = track_feats[indices]
            det_masks = det_masks[indices]
            a = 1
            len_thr = len(det_labels)
            height = batched_inputs[0]['height']
            width = batched_inputs[0]['width']
            bboxes, labels, ids, indices, masks_after_track = self.idol_tracker.match(
                bboxes=det_bboxes,
                labels=det_labels,
                masks=det_masks,
                track_feats=track_feats,
                frame_id=self.frame_id,
                indices=indices,
                datasets_ori=batched_inputs[0],
                ori_size=(height, width)
            )
        selected = torch.nonzero(ids >= 0).squeeze(-1)
        if selected.shape[0] != 0:
            N, C, H, W = masks_after_track.shape
            mask = F.interpolate(masks_after_track, size=(H * 4, W * 4), mode='bilinear', align_corners=False)
            mask = mask.sigmoid() > 0.5
            masks_after_track1 = mask[:, :, :image_size[0], :image_size[1]]
            mask = F.interpolate(masks_after_track1.float(), size=(height, width),
                                 mode='nearest')
            mask = mask.squeeze(1).byte()
            masks_after_track2 = mask
            rles = [
                mask_util.encode(np.array(mask1[:, :, None], order="F", dtype="uint8"))[0]
                for mask1 in masks_after_track2.cpu()
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            # mask_np = []

            for idx in selected:
                prop_idx = indices[idx]
                det = bboxes[idx]
                mask = rles[idx]
                x1, y1, x2, y2 = det[:4]
                w = int(x2 * batched_inputs[0]['width'])
                h = int(y2 * batched_inputs[0]['height'])
                c_x = int(x1 * batched_inputs[0]['width'])
                c_y = int(y1 * batched_inputs[0]['height'])
                x1 = max(0, int(c_x - w / 2))
                y1 = max(0, int(c_y - h / 2))
                x2 = int(c_x + w / 2)
                y2 = int(c_y + h / 2)

                prop = {
                    'bbox': [x1, y1, w, h],
                    'track_id': ids[idx].numpy().tolist(),
                    'category_id': 1,
                    'image_id': batched_inputs[0]['image_id'],
                    'video_id': video_id,
                    'score': float(det[-1]),
                    'segmentations': mask
                }
                self.tracklets1.append(prop)

            if self.tracklets1 != []:
                dst_root = '/ssd1/wby_workspace/VNext/1_track_result/tao/3_ioubranch_reiddepth/submit2/'
                if not os.path.exists(dst_root):
                    os.makedirs(dst_root)
                train_coco_path = dst_root + str(batched_inputs[0]['image_id']).zfill(10) + '.json'
                with open(train_coco_path, 'w') as f3:
                    json.dump(self.tracklets1, f3)
        self.tracklets1 = []
        self.frame_id += 1
        return processed_results



    def pseudo_boxes_merging_box_level(self, processed_results, box_cls, box_pred, mask_pred,
                                       reid_pred, iou_pred, box_i, datasets_ori, idx_ori, batched_inputs, item, image_size):
        logits = box_cls[0].sigmoid()
        # if iou_pred is not [None]:
        #     PBM_socre = [0.3, 0.3]
        logits = torch.sqrt(logits * iou_pred[0].sigmoid())
        # else:
        #     PBM_socre = [0.2, 0.15]
        # datasets_ori['annotations'] = []
        root = '/ssd1/wby_workspace/VNext/demo_img/'
        height_select = batched_inputs[0]['height']
        width_select = batched_inputs[0]['width']
        output_boxes = box_pred[0]
        output_mask = mask_pred[0]
        scores = logits.cpu().detach()
        max_score, _ = torch.max(logits, 1)
        indices = torch.nonzero(max_score > self.inference_select_thres, as_tuple=False).squeeze(1)  # 符合
        if len(indices) == 0:
            topkv, indices_top1 = torch.topk(scores.max(1)[0], k=1)
            indices_top1 = indices_top1[torch.argmax(topkv)]
            indices = [indices_top1.tolist()]
        else:
            nms_scores, idxs = torch.max(logits[indices], 1)
            boxes_before_nms = box_cxcywh_to_xyxy(output_boxes[indices])
            keep_indices = ops.batched_nms(boxes_before_nms, nms_scores, idxs, 0.9)  # .tolist()
            indices = indices[keep_indices]
        box_score = torch.max(logits[indices], 1)[0]
        det_bboxes = torch.cat([output_boxes[indices], box_score.unsqueeze(1)],
                               dim=1)
        det_labels = torch.argmax(logits[indices], dim=1)
        det_masks = output_mask[indices]
        valids = mask_nms1(det_masks, det_bboxes[:, -1], None, 0.3)
        det_bboxes_ori = det_bboxes[valids, :]
        det_masks = det_masks[valids, :]
        init_inds1 = det_bboxes_ori[:, 4] > 0.20
        det_bboxes_ori = det_bboxes_ori[init_inds1, :]  # ori pse
        det_masks1 = det_masks[init_inds1, :]
        init_inds = det_bboxes[:, 4] > 0.10
        det_bboxes = det_bboxes[init_inds, :]
        det_list = []
        det_list_ori = []
        for det in det_bboxes:
            i = det
            c_x = int(i[0] * width_select)
            c_y = int(i[1] * height_select)
            w = int(i[2] * width_select)
            h = int(i[3] * height_select)
            x1 = max(0, int(c_x - 0.5 * w))
            y1 = max(0, int(c_y - 0.5 * h))
            x2 = max(0, int(c_x + 0.5 * w))
            y2 = max(0, int(c_y + 0.5 * h))
            det_list.append([x1, y1, x2, y2, float(i[4]), h * w, int((x1 + x2) / 2), int((y1 + y2) / 2)])
        for det in det_bboxes_ori:
            i = det
            c_x = int(i[0] * width_select)
            c_y = int(i[1] * height_select)
            w = int(i[2] * width_select)
            h = int(i[3] * height_select)
            x1 = max(0, int(c_x - 0.5 * w))
            y1 = max(0, int(c_y - 0.5 * h))
            x2 = max(0, int(c_x + 0.5 * w))
            y2 = max(0, int(c_y + 0.5 * h))
            det_list_ori.append([x1, y1, x2, y2, float(i[4]), h * w, int((x1 + x2) / 2), int((y1 + y2) / 2)])

        sample = len(det_list)
        det_list.sort(key=lambda x: x[5], reverse=False)
        clusters = []
        for i in range(sample):
            cluster = []
            det1 = det_list[i]
            matched = 0
            if len(clusters) > 0:
                for idx, m in enumerate(clusters):
                    if i in m:
                        matched = 1
                        cluster = m
                        break
                if matched == 0:
                    cluster.append(i)
            else:
                cluster.append(i)
            for j in range(sample):
                if i == j or j in cluster:
                    continue
                det2 = det_list[j]
                iou = int(iou_score(det1[0], det1[1], det1[2], det1[3],
                                    det2[0], det2[1], det2[2], det2[3]) * 100)
                if iou > 70:
                    cluster.append(j)
            cluster = list(set(cluster))
            if matched == 1:
                clusters[idx] = cluster
            else:
                clusters.append(cluster)
        det_merge_single = []
        for idx1, m in enumerate(clusters):
            x1 = width_select
            x2 = 0
            y1 = height_select
            y2 = 0
            long_score = 0
            for n in m:
                x1 = min(x1, det_list[n][0])
                y1 = min(y1, det_list[n][1])
                x2 = max(x2, det_list[n][2])
                y2 = max(y2, det_list[n][3])
                long_score = float(max(long_score, det_list[n][4]))
            w = x2 - x1
            h = y2 - y1
            det_merge_single.append([x1, y1, x2, y2, long_score, w * h, int((x1 + x2) / 2), int((y1 + y2) / 2)])
        det_merge_single_nms = []
        sample = len(det_merge_single)
        for i in range(sample):
            matched = 0
            det_can = []
            det1 = det_merge_single[i]
            det_can.append(det1)
            for j in range(sample):
                if i == j:
                    continue
                det2 = det_merge_single[j]
                iou = int(iou_score(det1[0], det1[1], det1[2], det1[3],
                                    det2[0], det2[1], det2[2], det2[3]) * 100)
                if iou > 80:
                    matched = 1
                    det_can.append(det2)
            if matched == 1:
                x1 = width_select
                x2 = 0
                y1 = height_select
                y2 = 0
                long_score = 0
                for n in det_can:
                    x1 = min(x1, n[0])
                    y1 = min(y1, n[1])
                    x2 = max(x2, n[2])
                    y2 = max(y2, n[3])
                    long_score = float(max(long_score, n[4]))
                w = x2 - x1
                h = y2 - y1
                det_merge_single_nms.append(
                    [x1, y1, x2, y2, long_score, w * h, int((x1 + x2) / 2), int((y1 + y2) / 2)])
            else:
                det_merge_single_nms.append(det1)
        det_final = []
        det_list1_nms = []
        sample = len(det_merge_single_nms)
        sample1 = len(det_list_ori)
        for i in range(sample):
            matched = 0
            det_can = []
            det1 = det_merge_single_nms[i]
            det_can.append(det1)
            for j in range(sample1):
                det2 = det_list_ori[j]
                iou = int(iou_score(det1[0], det1[1], det1[2], det1[3],
                                    det2[0], det2[1], det2[2], det2[3]) * 100)
                if iou > 90:
                    matched = 1
                    det_can.append(det2)
                else:
                    det_list1_nms.append(det2)
            if matched == 1:
                x1 = width_select
                x2 = 0
                y1 = height_select
                y2 = 0
                long_score = 0
                for n in det_can:
                    x1 = min(x1, n[0])
                    y1 = min(y1, n[1])
                    x2 = max(x2, n[2])
                    y2 = max(y2, n[3])
                    long_score = float(max(long_score, n[4]))
                w = x2 - x1
                h = y2 - y1
                det_final.append([x1, y1, x2, y2, long_score, w * h, int((x1 + x2) / 2), int((y1 + y2) / 2)])
            else:
                det_final.append(det1)
        det_final = det_final + det_list1_nms

        det_final_nms = []
        sample = len(det_final)
        for i in range(sample - 1):
            det_nms = []
            det1 = det_final[i]
            if det1[0] == 0 and det1[2] == 0:
                continue
            matched = 0
            for j in range(i + 1, sample):
                det2 = det_final[j]
                iou = int(iou_score(det1[0], det1[1], det1[2], det1[3],
                                    det2[0], det2[1], det2[2], det2[3]) * 100)
                if iou > 70:
                    det_final[j] = [0, 0, 0, 0]
                    matched = 1
                    det_nms.append(det2)
            if matched == 1:
                x1 = width_select
                x2 = 0
                y1 = height_select
                y2 = 0
                long_score = 0
                for n in det_nms:
                    x1 = min(x1, n[0])
                    y1 = min(y1, n[1])
                    x2 = max(x2, n[2])
                    y2 = max(y2, n[3])
                    long_score = float(max(long_score, n[4]))
                w = x2 - x1
                h = y2 - y1
                det_final_nms.append([x1, y1, x2, y2, long_score, w * h, int((x1 + x2) / 2), int((y1 + y2) / 2)])
            else:
                det_final_nms.append(det1)

        for idx_pse, bbox in enumerate(det_final_nms):
            i = bbox
            if i[2]<1:
                c_x = int(i[0] * width_select)
                c_y = int(i[1] * height_select)
                w = int(i[2] * width_select)
                h = int(i[3] * height_select)
                x1 = max(0, int(c_x - 0.5 * w))
                y1 = max(0, int(c_y - 0.5 * h))
                x2 = max(0, int(c_x + 0.5 * w))
                y2 = max(0, int(c_y + 0.5 * h))
            else:
                x1 = int(i[0])
                y1 = int(i[1])
                x2 = int(i[2])
                y2 = int(i[3])
                w = x2-x1
                h = y2-y1
            if w<15 or h<15 or int(w*h)>int(width_select*height_select*0.7):
                continue
            tl_x = x1
            tl_y = y1
            self.anno.append({'file_name': batched_inputs[0]['file_name'],
                              'image_id': batched_inputs[0]['image_id'],
                              'segmentation': [
                                  [tl_x, tl_y, tl_x + w, tl_y, tl_x + w, tl_y + h, tl_x, tl_y + h]],
                              'bbox': [tl_x, tl_y, w, h],
                              'id': self.num_pse + int(int(box_i.device.index) + 1) * 100000000,
                              'category_id': 1,
                              'gen': 1,
                              'longscore': float(bbox[4]),
                              'iou_score': 0,
                              'iscrowd': 0,
                              'isgt_wby': 0,
                              'epoch0_gt': 1
                              })
            self.num_pse += 1

        if len(self.anno) > 0:
            dst_root = ''
            if not os.path.exists(dst_root):
                os.makedirs(dst_root)
            try:
                train_coco_path = dst_root + str(batched_inputs[0]['image_id']).zfill(10) + '.json'
                with open(train_coco_path, 'w') as f3:
                    json.dump(self.anno, f3)
            except:
                file = open(train_coco_path, 'w', encoding='utf-8');
                file.write(json.dumps(self.anno, cls=MyEncoder, indent=4))
        else:
            self.empty += 1
        self.anno = []

    def vis_boxes(self, datasets_ori, idx_ori, det_bboxes, width_select, height_select, img_dir_path):
        img_path = datasets_ori['file_name']
        img1 = cv2.imread('/ssd1/wby_workspace/VNext/' + img_path)
        for i in datasets_ori['annotations']:
            img1 = cv2.rectangle(img1, (int(i['bbox'][0]), int(i['bbox'][1])),
                          (int(i['bbox'][0] + i['bbox'][2]), int(i['bbox'][1] + i['bbox'][3])), (0, 255, 255), 2)
        for det in det_bboxes:
            i = det
            if i[2]<1 :
                c_x = int(i[0] * width_select)
                c_y = int(i[1] * height_select)
                w = int(i[2] * width_select)
                h = int(i[3] * height_select)
                x1 = max(0, int(c_x - 0.5 * w))
                y1 = max(0, int(c_y - 0.5 * h))
                x2 = max(0, int(c_x + 0.5 * w))
                y2 = max(0, int(c_y + 0.5 * h))
            else:
                x1 = int(i[0])
                y1 = int(i[1])
                x2 = int(i[2])
                y2 = int(i[3])
            img1 = cv2.rectangle(img1, (x1, y1), (x2, y2), (255, 0, 255), 1)
        if not os.path.exists(img_dir_path):
            os.makedirs(img_dir_path)
        cv2.imwrite(img_dir_path
            + str(idx_ori).zfill(6) + '_' + img_path.split('/')[-1], img1)
        a = 1

    def prepare_targets(self, targets, tags, only_det=False):
        if not only_det:
            new_targets = []
            truth_list = []
            for targets_per_image in targets:  # [instance_key, instance_ref, inst_key, ...]
                h, w = targets_per_image.image_size
                image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
                gt_classes = targets_per_image.gt_classes
                gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
                gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
                gt_masks = targets_per_image.gt_masks.tensor
                longscore = targets_per_image.longscore
                # gt_masks = []
                inst_ids = targets_per_image.gt_ids
                valid_id = inst_ids!=-1  # if a object is disappeared，its gt_ids is -1
                truth_list.append(valid_id)
                new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'masks': gt_masks, 'inst_id':inst_ids, 'valid':valid_id, 'longscore':longscore})
            bz = len(new_targets) // 2
            key_ids = list(range(0,bz*2-1,2))
            ref_ids = list(range(1,bz*2,2))
            det_targets = [new_targets[_i] for _i in key_ids]
            ref_targets = [new_targets[_i] for _i in ref_ids]
            for i in range(bz):
                det_target = det_targets[i]
                ref_target = ref_targets[i]
                if False in det_target['valid']:
                    valid_i = det_target['valid'].clone()
                    for k,v in det_target.items():
                        det_target[k] = v[valid_i]
                    for k,v in ref_target.items():
                        ref_target[k] = v[valid_i]
            tags1 = []
            return det_targets,ref_targets, tags1
        else:
            new_targets = []
            truth_list = []
            for targets_per_image in targets:
                h, w = targets_per_image.image_size
                image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
                gt_classes = targets_per_image.gt_classes
                gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
                gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
                gt_masks = targets_per_image.gt_masks.tensor
                longscore = targets_per_image.longscore
                # gt_masks = []
                inst_ids = targets_per_image.gt_ids
                valid_id = inst_ids != -1
                truth_list.append(valid_id)
                new_targets.append(
                    {"labels": gt_classes, "boxes": gt_boxes, 'masks': gt_masks, 'inst_id': inst_ids, 'valid': valid_id,
                     'longscore': longscore})

            tags1 = [[], []]
            for i in range(len(truth_list[0])):
                if truth_list[0][i] == True:
                    tags1[0].append(tags[0][i])
            for i in range(len(truth_list[1])):
                if truth_list[1][i]==True:
                    tags1[1].append(tags[1][i])
            bz = len(new_targets) // 2
            key_ids = list(range(0, bz * 2 - 1, 2))
            det_targets = [new_targets[_i] for _i in key_ids]
            for i in range(bz):  # fliter empety object in key frame
                det_target = det_targets[i]
                if False in det_target['valid']:
                    valid_i = det_target['valid'].clone()
                    for k, v in det_target.items():
                        det_target[k] = v[valid_i]

            return det_targets, [], tags1

    def inference(self, outputs, tracker, ori_size, image_sizes, datasets_ori):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        # results = []
        video_dict = {}
        vido_logits = outputs['pred_logits']
        video_output_masks = outputs['pred_masks']
        output_h, output_w = video_output_masks.shape[-2:]
        video_output_boxes = outputs['pred_boxes']
        video_output_embeds = outputs['pred_inst_embed']
        vid_len = len(vido_logits)
        for i_frame, (logits, output_mask, output_boxes, output_embed) in enumerate(zip(
            vido_logits, video_output_masks, video_output_boxes, video_output_embeds
         )):
            i_frame = self.img_count
            scores = logits.sigmoid().cpu().detach()
            max_score, _ = torch.max(logits.sigmoid(),1)
            indices = torch.nonzero(max_score>self.inference_select_thres, as_tuple=False).squeeze(1)  # 符合
            if len(indices) == 0:
                topkv, indices_top1 = torch.topk(scores.max(1)[0],k=1)
                indices_top1 = indices_top1[torch.argmax(topkv)]
                indices = [indices_top1.tolist()]
            else:
                nms_scores,idxs = torch.max(logits.sigmoid()[indices],1)
                boxes_before_nms = box_cxcywh_to_xyxy(output_boxes[indices])
                keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,0.9)#.tolist()
                indices = indices[keep_indices]
            box_score = torch.max(logits.sigmoid()[indices],1)[0]
            det_bboxes = torch.cat([output_boxes[indices],box_score.unsqueeze(1)],dim=1)
            det_labels = torch.argmax(logits.sigmoid()[indices],dim=1)
            track_feats = output_embed[indices]
            det_masks = output_mask[indices]
            bboxes, labels, ids, indices = self.idol_tracker.match(
                bboxes=det_bboxes,
                labels=det_labels,
                masks = det_masks,
                track_feats=track_feats,
                frame_id=i_frame,
                indices = indices,
                datasets_ori=datasets_ori,
                ori_size=ori_size
            )
    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(self.normalizer(frame.to(self.device)))
        images = ImageList.from_tensors(images)
        return images


    def coco_inference(self, box_cls, box_pred, mask_pred, image_sizes, reid_pred=None, iou_pred=None):
      
        assert len(box_cls) == len(image_sizes)
        results = []

        for i, (logits_per_image, box_pred_per_image, image_size, iou_per_image) in enumerate(zip(
            box_cls, box_pred, image_sizes, iou_pred
        )):
        # for i, (logits_per_image, box_pred_per_image, image_size) in enumerate(zip(
        #         box_cls, box_pred, image_sizes
        # )):
            prob = logits_per_image.sigmoid()
            if iou_per_image is not None:
                prob = torch.sqrt(prob * iou_per_image.sigmoid())
            nms_scores,idxs = torch.max(prob,1)
            boxes_before_nms = box_cxcywh_to_xyxy(box_pred_per_image)
            keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,0.7)  
            prob = prob[keep_indices]
            box_pred_per_image = box_pred_per_image[keep_indices]
            mask_pred_i = mask_pred[i][keep_indices]

            if not self.only_det:
                reid_pred_i = reid_pred[i][keep_indices]
            else:
                reid_pred_i = None
            topk_num = min(100, prob.shape[0])
            topk_values, topk_indexes = torch.topk(prob.view(-1), topk_num, dim=0)
            scores = topk_values
            topk_boxes = torch.div(topk_indexes, logits_per_image.shape[1], rounding_mode='floor')
            labels = topk_indexes % logits_per_image.shape[1]
            scores_per_image = scores
            labels_per_image = labels

            box_pred_per_image = box_pred_per_image[topk_boxes]
            mask_pred_i = mask_pred_i[topk_boxes]
            # [CHANGE]
            if not self.only_det:
                reid_pred_i = reid_pred_i[topk_boxes]
            else:
                reid_pred_i = None
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                N, C, H, W = mask_pred_i.shape
                mask = F.interpolate(mask_pred_i, size=(H*4, W*4), mode='bilinear', align_corners=False)
                mask = mask.sigmoid() > 0.5
                mask = mask[:,:,:image_size[0],:image_size[1]]
                result.pred_masks = mask
            if not self.only_det:
                result.embeddings = reid_pred_i
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        if self.only_det:
            reid_pred_i = None

        return results, box_pred_per_image, mask_pred_i, reid_pred_i



    def preprocess_coco_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        try:
            images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        except:
            images = [self.normalizer(x["image"].to(self.device)) for x in [batched_inputs]]
        images = ImageList.from_tensors(images)
        return images



    
