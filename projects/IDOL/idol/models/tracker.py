# ------------------------------------------------------------------------
# IDOL: In Defense of Online Models for Video Instance Segmentation
# Copyright (c) 2022 ByteDance. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import torchvision.ops as ops
import numpy as np
from detectron2.structures import Boxes, ImageList, Instances, BitMasks


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


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


def mask_iou(mask1, mask2):
    # print('mask1.shape = {}, mask2.shape = {}'.format(mask1.shape, mask2.shape))
    mask1 = mask1.char()
    mask2 = mask2.char()

    intersection = (mask1[:, :, :] * mask2[:, :, :]).sum(-1).sum(-1)
    union = (mask1[:, :, :] + mask2[:, :, :] - mask1[:, :, :] * mask2[:, :, :]).sum(-1).sum(-1)

    return (intersection + 1e-6) / (union + 1e-6)


def mask_nms(seg_masks, scores, category_ids, nms_thr=0.5):
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


class IDOL_Tracker(object):

    def __init__(self,
                 nms_thr_pre=0.7,
                 nms_thr_post=0.3,
                 init_score_thr=0.2,
                 addnew_score_thr=0.5,
                 obj_score_thr=0.1,
                 match_score_thr=0.5,
                 memo_tracklet_frames=10,
                 memo_backdrop_frames=1,
                 memo_momentum=0.5,
                 nms_conf_thr=0.5,
                 nms_backdrop_iou_thr=0.5,
                 nms_class_iou_thr=0.7,
                 with_cats=True,
                 match_metric='bisoftmax',
                 long_match=False,
                 frame_weight=False,
                 temporal_weight=False,
                 memory_len=10):
        assert 0 <= memo_momentum <= 1.0
        assert memo_tracklet_frames >= 0
        assert memo_backdrop_frames >= 0
        self.memory_len = memory_len
        self.temporal_weight = temporal_weight
        self.long_match = long_match
        self.frame_weight = frame_weight
        self.nms_thr_pre = nms_thr_pre
        self.nms_thr_post = nms_thr_post
        self.init_score_thr = init_score_thr
        self.addnew_score_thr = addnew_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.memo_tracklet_frames = memo_tracklet_frames
        self.memo_backdrop_frames = memo_backdrop_frames
        self.memo_momentum = memo_momentum
        self.nms_conf_thr = nms_conf_thr
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.nms_class_iou_thr = nms_class_iou_thr
        self.with_cats = with_cats
        assert match_metric in ['bisoftmax', 'softmax', 'cosine']
        self.match_metric = match_metric

        self.num_tracklets = 0
        self.tracklets = dict()
        self.backdrops = []
        self.num_gt = 0
        self.num_pse = 0
        self.num_pse2 = 0
        self.bbox_gen = []
        self.bbox_gen2 = []

    @property
    def empty(self):
        return False if self.tracklets else True

    def iou_match_grid(self, box1, box2):

        x11, y11, x12, y12 = np.split(box1, 4, axis=1)
        x21, y21, x22, y22 = np.split(box2, 4, axis=1)

        xa = np.maximum(x11, np.transpose(x21))
        xb = np.minimum(x12, np.transpose(x22))
        ya = np.maximum(y11, np.transpose(y21))
        yb = np.minimum(y12, np.transpose(y22))

        area_inter = np.maximum(0, (xb - xa + 1)) * np.maximum(0, (yb - ya + 1))

        area_1 = (x12 - x11 + 1) * (y12 - y11 + 1)
        area_2 = (x22 - x21 + 1) * (y22 - y21 + 1)
        area_union = area_1 + np.transpose(area_2) - area_inter

        iou = area_inter / area_union

        return iou

    def iou_socre(self, x1, y1, x2, y2, a1, b1, a2, b2):
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

    def update_memo(self, ids, bboxes, embeds, labels, frame_id, datasets_ori, ori_size, conf_list):
        tracklet_inds = ids > -1

        # update memo
        num_exist1 = 0
        num_new1 = 0
        anno_gt = []
        self.num_gt += len(anno_gt)
        for idx1, (id, bbox, embed, label) in enumerate(zip(ids[tracklet_inds],
                                                            bboxes[tracklet_inds],
                                                            embeds[tracklet_inds],
                                                            labels[tracklet_inds])):
            conf = conf_list[idx1]
            id = int(id)
            if id in self.tracklets.keys():
                num_exist1 += 1
                if self.tracklets[id]['exist_frame'] == 1:
                    velocity = (bbox - self.tracklets[id]['bbox']) / (
                            frame_id - self.tracklets[id]['last_frame'])
                elif self.tracklets[id]['exist_frame'] > 1:
                    velocity = (bbox - self.tracklets[id]['bbox'][-1]) / (
                            frame_id - self.tracklets[id]['last_frame'])

                self.tracklets[id]['bbox'] = bbox
                self.tracklets[id]['long_score'].append(bbox[-1])
                self.tracklets[id]['embed'] = (
                                                      1 - self.memo_momentum
                                              ) * self.tracklets[id]['embed'] + self.memo_momentum * embed
                self.tracklets[id]['long_embed'].append(embed)
                # self.tracklets[id]['long_embed'].append(self.tracklets[id]['embed'])
                self.tracklets[id]['last_frame'] = frame_id
                self.tracklets[id]['label'] = label
                self.tracklets[id]['velocity'] = (
                                                         self.tracklets[id]['velocity'] *
                                                         self.tracklets[id]['acc_frame'] + velocity) / (
                                                         self.tracklets[id]['acc_frame'] + 1)
                self.tracklets[id]['acc_frame'] += 1
                self.tracklets[id]['exist_frame'] += 1

                # if self.tracklets[id]['exist_frame'] >= 1:
                #     width = ori_size[1]
                #     height = ori_size[0]
                #     i = bbox
                #     c_x = int(i[0] * width)
                #     c_y = int(i[1] * height)
                #     w = int(i[2] * width)
                #     h = int(i[3] * height)
                #     tl_x = int(c_x - 0.5 * w)
                #     tl_y = int(c_y - 0.5 * h)
                #     br_x = int(c_x + 0.5 * w)
                #     br_y = int(c_y + 0.5 * h)
                #     r1 = max(float(i[3]/i[2]), float(i[2]/i[3]))
                #     size = h * w
                #     # if r1 <= 10 and size > 100 and size < int((width * height * 0.4)):
                #     # if r1 <= 10:
                #     label_bbox = int(self.tracklets[id]['label'])
                #     # self.bbox_gen.append({'segmentation': [[tl_x, tl_y, tl_x+w, tl_y, tl_x+w, tl_y+h, tl_x, tl_y+h]],
                #     #                       'bbox': [tl_x, tl_y, w, h],
                #     #                       'id': -1,
                #     #                       'category_id': 1,
                #     #                       'gen': 1,
                #     #                       'label': label_bbox,
                #     #                       'longscore': float(bbox[-1]),
                #     #                       'iou_score': 0,
                #     #                       'iscrowd':0,
                #     #                       'area': int(w*h),
                #     #                       'file_name':datasets_ori['file_name']
                #     #                       })
                #     self.bbox_gen.append({'file_name': datasets_ori['file_name'].replace(
                #         'datasets/UVO/uvo_videos_dense_frames/datasets/UVO/uvo_videos_dense_frames/',
                #         'datasets/UVO/uvo_videos_dense_frames/'),
                #         'image_id': datasets_ori['image_id'],
                #         'segmentation': [[tl_x, tl_y, tl_x + w, tl_y, tl_x + w, tl_y + h, tl_x, tl_y + h]],
                #         # 'segmentation': rles[idx_pse],
                #         'bbox': [tl_x, tl_y, w, h],
                #         # 'id': self.num_pse + int(idx_ori%(len_datasets/num_gpu) + 1)*100000000,
                #         # 'id': self.num_pse + int(int(bboxes.device.index) + 1) * 10000000000,
                #         'id': self.num_pse,
                #         'category_id': 1,
                #         'gen': 1,
                #         'longscore': float(bbox[-1]),
                #         # 'longscore': 1,
                #         'iou_score': 1,
                #         'iscrowd': 0,
                #         'isgt_wby': 0,
                #         'epoch0_gt': 1,
                #         'track_conf': round(float(conf), 4)
                #     })
                #     # print(int(bboxes.device.index))
                #     self.num_pse += 1
                #     a = 1
                #     # img = cv2.rectangle(img, (tl_x, tl_y),
                #     #               (br_x, br_y), (255, 255, 0), 1)
                #     # anno_gt = datasets_ori['annotations']
                #     # bbox_score = []
                #     # for i in anno_gt:
                #     #     gt = i['bbox']
                #     #     iou_score = self.iou_match_grid(bbox, gt)
                #     #     bbox_score.append(iou_score)
                # if self.tracklets[id]['exist_frame'] > 1:
                #     width = ori_size[1]
                #     height = ori_size[0]
                #     i = bbox
                #     c_x = int(i[0] * width)
                #     c_y = int(i[1] * height)
                #     w = int(i[2] * width)
                #     h = int(i[3] * height)
                #     tl_x = int(c_x - 0.5 * w)
                #     tl_y = int(c_y - 0.5 * h)
                #     br_x = int(c_x + 0.5 * w)
                #     br_y = int(c_y + 0.5 * h)
                #     r1 = max(float(i[3]/i[2]), float(i[2]/i[3]))
                #     if len(anno_gt)>0:
                #         iou_score = []
                #         for j in anno_gt:
                #             j_tl_x = j['bbox'][0]
                #             j_tl_y = j['bbox'][1]
                #             j_br_x = j['bbox'][2] + j['bbox'][0]
                #             j_br_y = j['bbox'][3] + j['bbox'][1]
                #             iou_score.append(self.iou_socre(tl_x, tl_y, br_x, br_y,
                #                                        j_tl_x, j_tl_y, j_br_x, j_br_y)*100)
                #         iou_score = int(max(iou_score))
                #         size = h * w
                #         if iou_score<30:
                #             label_bbox = int(self.tracklets[id]['label'])
                #             self.bbox_gen.append({'segmentation': [[tl_x, tl_y, tl_x+w, tl_y, tl_x+w, tl_y+h, tl_x, tl_y+h]],
                #                                  'bbox': [tl_x, tl_y, w, h],
                #                                  'id': -1,
                #                                  'category_id': 1,
                #                                  'gen': 1,
                #                                  'label': label_bbox,
                #                                  'longscore': float(bbox[-1]),
                #                                  'iou_score': iou_score,
                #                                  'iscrowd':0,
                #                                  'area': int(w*h),
                #                                  'file_name':datasets_ori['file_name']
                #             })
                #             # self.num_pse += 1
                #             # img = cv2.rectangle(img, (tl_x, tl_y),
                #             #               (br_x, br_y), (255, 255, 0), 1)
                #     else:
                #         size = h * w
                #         # if r1 <= 10 and size > 100 and size < int((width * height * 0.4)):
                #         # if r1 <= 10:
                #         label_bbox = int(self.tracklets[id]['label'])
                #         # self.bbox_gen.append({'segmentation': [[tl_x, tl_y, tl_x+w, tl_y, tl_x+w, tl_y+h, tl_x, tl_y+h]],
                #         #                       'bbox': [tl_x, tl_y, w, h],
                #         #                       'id': -1,
                #         #                       'category_id': 1,
                #         #                       'gen': 1,
                #         #                       'label': label_bbox,
                #         #                       'longscore': float(bbox[-1]),
                #         #                       'iou_score': 0,
                #         #                       'iscrowd':0,
                #         #                       'area': int(w*h),
                #         #                       'file_name':datasets_ori['file_name']
                #         #                       })
                #         self.bbox_gen2.append({'file_name': datasets_ori['file_name'].replace(
                #             'datasets/UVO/uvo_videos_dense_frames/datasets/UVO/uvo_videos_dense_frames/',
                #             'datasets/UVO/uvo_videos_dense_frames/'),
                #             'image_id': datasets_ori['image_id'],
                #             'segmentation': [[tl_x, tl_y, tl_x + w, tl_y, tl_x + w, tl_y + h, tl_x, tl_y + h]],
                #             # 'segmentation': rles[idx_pse],
                #             'bbox': [tl_x, tl_y, w, h],
                #             # 'id': self.num_pse + int(idx_ori%(len_datasets/num_gpu) + 1)*100000000,
                #             'id': self.num_pse2 + int(int(bboxes.device.index) + 1) * 10000000000,
                #             'category_id': 1,
                #             'gen': 1,
                #             'longscore': float(bbox[-1]),
                #             # 'longscore': 1,
                #             'iou_score': 1,
                #             'iscrowd': 0,
                #             'isgt_wby': 0,
                #             'epoch0_gt': 1
                #         })
                #         # print(int(bboxes.device.index))
                #         self.num_pse2 += 1
                #         # self.num_pse += 1
                #         a = 1
                #         # img = cv2.rectangle(img, (tl_x, tl_y),
                #         #               (br_x, br_y), (255, 255, 0), 1)
                #     # anno_gt = datasets_ori['annotations']
                #     # bbox_score = []
                #     # for i in anno_gt:
                #     #     gt = i['bbox']
                #     #     iou_score = self.iou_match_grid(bbox, gt)
                #     #     bbox_score.append(iou_score)
            else:
                num_new1 += 1
                self.tracklets[id] = dict(
                    bbox=bbox,
                    # bbox_all=bbox.cpu().numpy(),
                    embed=embed,
                    long_embed=[embed],
                    long_score=[bbox[-1]],
                    label=label,
                    last_frame=frame_id,
                    velocity=torch.zeros_like(bbox),
                    acc_frame=0,
                    exist_frame=1)
        backdrop_inds = torch.nonzero(ids == -1, as_tuple=False).squeeze(1)
        self.backdrops.insert(
            0,
            dict(
                bboxes=bboxes[backdrop_inds],
                embeds=embeds[backdrop_inds],
                labels=labels[backdrop_inds]))

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if frame_id - v['last_frame'] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
            if len(v['long_embed']) > self.memory_len:
                v['long_embed'].pop(0)
            if len(v['long_score']) > self.memory_len:
                v['long_score'].pop(0)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

        if len(self.backdrops) > self.memo_backdrop_frames:
            self.backdrops.pop()

    @property
    def memo(self):
        memo_embeds = []
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_vs = []
        memo_long_embeds = []
        memo_long_score = []
        memo_exist_frame = []
        for k, v in self.tracklets.items():
            memo_bboxes.append(v['bbox'][None, :])
            if self.long_match:
                weights = torch.stack(v['long_score'])
                if self.temporal_weight:
                    length = len(weights)
                    temporal_weight = torch.range(0.0, 1, 1 / length)[1:].to(weights)
                    weights = weights + temporal_weight
                sum_embed = (torch.stack(v['long_embed']) * weights.unsqueeze(1)).sum(0) / weights.sum()
                memo_embeds.append(sum_embed[None, :])
            else:
                memo_embeds.append(v['embed'][None, :])

            memo_long_embeds.append(torch.stack(v['long_embed']))
            memo_long_score.append(torch.stack(v['long_score']))
            memo_exist_frame.append(v['exist_frame'])
            memo_ids.append(k)
            memo_labels.append(v['label'].view(1, 1))
            memo_vs.append(v['velocity'][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)
        memo_exist_frame = torch.tensor(memo_exist_frame, dtype=torch.long)

        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)

        memo_vs = torch.cat(memo_vs, dim=0)
        return memo_bboxes, memo_labels, memo_embeds, memo_ids.squeeze(
            0), memo_vs, memo_long_embeds, memo_long_score, memo_exist_frame

    def match(self, bboxes, labels, masks, track_feats, frame_id, indices, datasets_ori=[], ori_size=[]):

        embeds = track_feats

        #   mask nms
        valids = mask_nms(masks, bboxes[:, -1], None, self.nms_thr_pre)

        # print('end--------')
        mask_new_indices = torch.tensor(indices)[valids].tolist()  # list
        indices = mask_new_indices
        bboxes = bboxes[valids, :]
        labels = labels[valids]
        masks = masks[valids]
        embeds = embeds[valids, :]
        ids = torch.full((bboxes.size(0),), -2, dtype=torch.long)  # [num_bboxes]

        # match if buffer is not empty
        if bboxes.size(0) > 0 and not self.empty:
            # print('full')
            (memo_bboxes, memo_labels, memo_embeds, memo_ids,
             memo_vs, memo_long_embeds, memo_long_score, memo_exist_frame) = self.memo
            # print(memo_embeds.shape)
            memo_exist_frame = memo_exist_frame.to(memo_embeds)
            memo_ids = memo_ids.to(memo_embeds)
            if self.match_metric == 'longrang':
                feats = torch.mm(embeds, memo_embeds.t())
            elif self.match_metric == 'bisoftmax':
                feats = torch.mm(embeds, memo_embeds.t())
                d2t_scores = feats.softmax(dim=1)
                t2d_scores = feats.softmax(dim=0)
                scores = (d2t_scores + t2d_scores) / 2
            elif self.match_metric == 'softmax':
                feats = torch.mm(embeds, memo_embeds.t())
                scores = feats.softmax(dim=1)
            elif self.match_metric == 'cosine':
                scores = torch.mm(
                    F.normalize(embeds, p=2, dim=1),
                    F.normalize(memo_embeds, p=2, dim=1).t())
            else:
                raise NotImplementedError

            # conf_list = torch.zeros_like(ids).to(bboxes.device)
            conf_list = []
            for i in range(bboxes.size(0)):
                if self.frame_weight:
                    non_backs = (memo_ids > -1) & (scores[i, :] > 0.5)
                    if (scores[i, non_backs] > 0.5).sum() > 1:
                        wighted_scores = scores.clone()
                        frame_weight = memo_exist_frame[scores[i, :][memo_ids > -1] > 0.5]
                        wighted_scores[i, non_backs] = wighted_scores[i, non_backs] * frame_weight
                        wighted_scores[i, ~non_backs] = wighted_scores[i, ~non_backs] * frame_weight.mean()
                        conf, memo_ind = torch.max(wighted_scores[i, :], dim=0)
                    else:
                        conf, memo_ind = torch.max(scores[i, :], dim=0)
                else:
                    conf, memo_ind = torch.max(scores[i, :], dim=0)
                id = memo_ids[memo_ind]
                conf_list.append(conf)
                if conf > self.match_score_thr:
                    if id > -1:
                        ids[i] = id
                        scores[:i, memo_ind] = 0
                        scores[i + 1:, memo_ind] = 0
            new_inds = (ids == -2) & (bboxes[:, 4] > self.addnew_score_thr).cpu()
            num_news = new_inds.sum()
            ids[new_inds] = torch.arange(
                self.num_tracklets,
                self.num_tracklets + num_news,
                dtype=torch.long)
            self.num_tracklets += num_news

            unselected_inds = torch.nonzero(ids == -2, as_tuple=False).squeeze(1)
            if len(unselected_inds) > 0:
                a = 1
            mask_ious = mask_iou(masks[unselected_inds].sigmoid() > 0.5, masks.permute(1, 0, 2, 3).sigmoid() > 0.5)
            #
            for i, ind in enumerate(unselected_inds):
                if (mask_ious[i, :ind] < self.nms_thr_post).all():
                    ids[ind] = -1
            self.update_memo(ids, bboxes, embeds, labels, frame_id, datasets_ori, ori_size, conf_list)


        elif self.empty:
            # print('empty')
            conf_list = bboxes[:, 4]
            for idx, item1 in enumerate(conf_list):
                conf_list[idx] = 0.7501
            init_inds = (ids == -2) & (bboxes[:, 4] > self.init_score_thr).cpu()
            num_news = init_inds.sum()
            ids[init_inds] = torch.arange(
                self.num_tracklets,
                self.num_tracklets + num_news,
                dtype=torch.long)
            self.num_tracklets += num_news
            unselected_inds = torch.nonzero(ids == -2, as_tuple=False).squeeze(1)
            mask_ious = mask_iou(masks[unselected_inds].sigmoid() > 0.5, masks.permute(1, 0, 2, 3).sigmoid() > 0.5)
            for i, ind in enumerate(unselected_inds):
                if (mask_ious[i, :ind] < self.nms_thr_post).all():
                    ids[ind] = -1
            self.update_memo(ids, bboxes, embeds, labels, frame_id, datasets_ori, ori_size, conf_list)

        return bboxes, labels, ids, indices, masks
        # return bboxes, labels, ids, indices

