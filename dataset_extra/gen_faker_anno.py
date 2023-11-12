import os
import json
from re import L
import numpy as np
from collections import defaultdict
from tkinter.font import families

def cal_iou(box1, box2):
    """
    :param box1: = [left1, top1, right1, bottom1]
    :param box2: = [left2, top2, right2, bottom2]
    :return: 
    """
    # left1, top1, right1, bottom1 = box1
    # left2, top2, right2, bottom2 = box2
    left1, top1, w1, h1 = box1
    left2, top2, w2, h2 = box2
    right1 = left1 + w1
    bottom1 = top1 + h1
    right2 = left2 + w2
    bottom2 = top2 + h2
    # 计算每个矩形的面积
    s1 = (right1 - left1) * (bottom1 - top1)  # b1的面积
    s2 = (right2 - left2) * (bottom2 - top2)  # b2的面积
 
    # 计算相交矩形
    left = max(left1, left2)
    top = max(top1, top2)
    right = min(right1, right2)
    bottom = min(bottom1, bottom2)
 
    # 相交框的w,h
    w = max(0, right - left)
    h = max(0, bottom - top)
    a1 = w * h  # C∩G的面积
    a2 = s1 + s2 - a1
    if a2 == 0:
        return -1
    iou = a1 / a2 #iou = a1/ (s1 + s2 - a1)
    return iou
    

def iou_match_grid(box1, box2):
 
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

def gen():
    prjson = 'datasets/props_train.json'  # 这个官方80类结果改编的
    save_path = 'datasets/tao/annotations/faker_tao_train_agn.json'
    with open(prjson, 'r') as f:
        dataset = json.load(f)

    imgs = {}
    for img_info in dataset['images']:
        imgs[img_info['id']] = img_info
    for anno in dataset['annotations']:  # 补充一个frame_index
        anno['frame_index'] = imgs[anno['image_id']]['frame_index']
    # 计算一下间隔
    vid2imgs = defaultdict(list)
    for img_info in dataset['images']:
        vid = img_info['video_id']
        vid2imgs[vid].append(img_info)
    vid2gap = {}
    for k,v in vid2imgs.items():
        v = sorted(v, key=lambda x: x['frame_index'])
        if len(v) > 1:
            vid2gap[k] = v[1]['frame_index'] - v[0]['frame_index']
        else:  # 如果只有一张图那么轨迹长度也只有1 后面用不到
            vid2gap[k] = 0
    #
    vtracks = {}
    for anno in dataset['annotations']:
        vid = anno['video_id']
        tid = anno['track_id']
        vt_key = '{}_{}'.format(vid, tid)
        if vt_key not in vtracks:
            vtracks[vt_key] = {
                'video_id': vid,
                'track_id': tid,
                'count': 1,
                'annos': [anno]
            }
        else:
            vtracks[vt_key]['count'] += 1  # [TODO]  这是总长度 不是连续长度
            vtracks[vt_key]['annos'].append(anno)
    # 计算轨迹稳定性
    for k,track in vtracks.items():
        vid = track['video_id']
        gap = vid2gap[vid]
        annos = track['annos']
        annos = sorted(annos, key=lambda x: x['frame_index'])
        if len(annos) == 1:
            track['score'] = 1
            annos[0]['ccnt'] = 1
        else:
            continus_cnt = 1
            max_cocnt = 1
            for i in range(1, len(annos)):
                iou = cal_iou(annos[i]['bbox'], annos[i-1]['bbox'])  # 不同视频运动速率不太一样，所以不是很好找
                if annos[i]['frame_index'] - annos[i-1]['frame_index'] == gap and iou > 0.5:  # [TODO] 可以加额外条件
                # if iou > 0.5:  # train set中有些不是间隔30f e.g. vid_2132 效果也还是可以的
                    continus_cnt += 1
                else:
                    # --------
                    idx = i - 1
                    while idx >= 0:
                        if annos[idx].get('ccnt', None) is not None:
                            break
                        annos[idx]['ccnt'] = continus_cnt
                        idx -= 1
                    # ---------
                    continus_cnt = 1
                max_cocnt = max(max_cocnt, continus_cnt)
            # ---------
            idx = len(annos) -1
            while idx >= 0:
                if annos[idx].get('ccnt', None) is not None:
                    break
                annos[idx]['ccnt'] = continus_cnt
                idx -= 1
            # ----------
            track['score'] = max_cocnt
            

    new_annotations = []
    for anno in dataset['annotations']:
        vid = anno['video_id']
        tid = anno['track_id']
        vt_key = '{}_{}'.format(vid, tid)
        # [TODO] 在此处针对每个box的尺寸比例作额外限制
        x1, y1, w, h = anno['bbox']
        w = max(0.00001, w)
        h = max(0.00001, h)
        ratio = max(1.0 * w / h, 1.0 * h / w)
        area = anno.get('area', w * h)
        img_id = anno['image_id']
        img_w, img_h = imgs[img_id]['width'], imgs[img_id]['height']
        img_size = img_w * img_h
        mix = area / img_size  # 占比
        if vtracks[vt_key]['score'] >= 5 and ratio < 4 and area > 100 and mix < 0.4 and anno.get('ccnt') >= 5:
            new_annotations.append(anno)
    dataset['annotations'] = new_annotations
    print('faker_annos_num = {}'.format(len(new_annotations)))
    imgids = list(set([anno['image_id'] for anno in new_annotations]))
    print('len_imgids = {}'.format(len(imgids)))

    # 提取gt_data进行合并
    gt_path = '/home/dhw/LTL_workspace/VNext/datasets/tao/annotations/train.json'
    with open(gt_path, 'r') as f:
        gtdata = json.load(f)
    knowns = {4, 13, 1038, 544, 1057, 34, 35, 36, 41, 45, 58, 60, 579, 1091, 1097, 1099, 78, 79, 81, 91, 1115,
                    1117, 95, 1122, 99, 1132, 621, 1135, 625, 118, 1144, 126, 642, 1155, 133, 1162, 139, 154, 174, 185,
                    699, 1215, 714, 717, 1229, 211, 729, 221, 229, 747, 235, 237, 779, 276, 805, 299, 829, 852, 347,
                    371, 382, 896, 392, 926, 937, 428, 429, 961, 452, 979, 980, 982, 475, 480, 993, 1001, 502, 1018}
    gtdata['annotations'] = [anno for anno in gtdata['annotations'] if anno['category_id'] in knowns]
    gt_img_ids = list(set([anno['image_id'] for anno in gtdata['annotations']]))
    gtdata['images'] = [img_info for img_info in gtdata['images'] if img_info['id'] in gt_img_ids]
    gt_img2annos = defaultdict(list)
    for anno in gtdata['annotations']:
        gt_img2annos[anno['image_id']].append(anno)

    fa_img_ids = list(set([anno['image_id'] for anno in dataset['annotations']]))
    fa_img_ids = [img_id for img_id in fa_img_ids if img_id in gt_img_ids]
    fa_img2annos = defaultdict(list)
    for anno in dataset['annotations']:
        fa_img2annos[anno['image_id']].append(anno)
    # 合并两个数据集 保留gt iou过滤生成的
    new_annos = []
    max_gt_anno_id = max([anno['id'] for anno in gtdata['annotations']]) + 1
    for img_id, gt_annos in gt_img2annos.items():
        fa_annos = fa_img2annos.get(img_id, [])
        if len(fa_annos) > 0:
            gtbboxes, fabboxes = [], []
            for anno in gt_annos:
                x, y, w, h = anno['bbox']
                gtbboxes.append([x, y, x + w, y + h])
            for anno in fa_annos:
                x, y, w, h = anno['bbox']
                fabboxes.append([x, y, x + w, y + h])
            gtbboxes = np.array(gtbboxes)
            fabboxes = np.array(fabboxes)
            # print('gt = {}, fa = {}'.format(gtbboxes.shape, fabboxes.shape))
            ious = iou_match_grid(fabboxes, gtbboxes)
            # print(ious)
            inds = np.where(ious.max(axis=1) < 0.5)[0]
            # print(inds)
            for idx in inds:
                new_anno = fa_annos[idx]
                new_anno['id'] = max_gt_anno_id
                # new_anno['segmentation'] = None  # 不能为None 否则Load_as_coco里面obj没有seg
                x1, y1, w, h = new_anno['bbox']
                x2, y2 = x1 + w, y1 + h
                new_anno['segmentation'] = [[x1, y1, x2, y1, x2, y2, x1, y2]]
                max_gt_anno_id += 1
                new_annos.append(new_anno)
    gtdata['annotations'].extend(new_annos)
    print("filter_gt_new_annos = {}".format(len(new_annos)))
    # 修改为 cls_agn
    gtdata['categories'] = [
        {'frequency': 'c', 'id': 1, 'synset': 'anklet.n.03', 'image_count': 1, 'instance_count': 1, 'name': 'coco'}
    ]
    for anno in gtdata['annotations']:
        anno['category_id'] = 1
    assert len(set([anno['category_id'] for anno in gtdata['annotations']])) == 1
            
    # with open(save_path, 'w') as f:
    #     f.write(json.dumps(gtdata))

# 根据 track_result得到伪标签json


# 查看一些gt_data里面的数据量
def view_tao_json():
    json_path = '/home/dhw/LTL_workspace/VNext/datasets/tao/annotations/train.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    imgids = list(set([anno['image_id'] for anno in data['annotations']]))
    print('len_imgids = {}, len_images = {}'.format(len(imgids), len(data['images'])))  # len_imgids = 17209, len_images = 18274

    knowns = {4, 13, 1038, 544, 1057, 34, 35, 36, 41, 45, 58, 60, 579, 1091, 1097, 1099, 78, 79, 81, 91, 1115,
                    1117, 95, 1122, 99, 1132, 621, 1135, 625, 118, 1144, 126, 642, 1155, 133, 1162, 139, 154, 174, 185,
                    699, 1215, 714, 717, 1229, 211, 729, 221, 229, 747, 235, 237, 779, 276, 805, 299, 829, 852, 347,
                    371, 382, 896, 392, 926, 937, 428, 429, 961, 452, 979, 980, 982, 475, 480, 993, 1001, 502, 1018}
    print('num_known_cates = {}'.format(len(knowns)))
    known_imgids = list(set([anno['image_id'] for anno in data['annotations'] if anno['category_id'] in knowns]))  # len_known_imgs = 16141
    print('len_known_imgs = {}'.format(len(known_imgids)))  # len_known_imgs = 16141
    known_annos = [anno for anno in data['annotations'] if anno['category_id'] in knowns]
    print('len_known_annos = {}'.format(len(known_annos)))  # len_known_annos = 43380


if __name__ == '__main__':
    view_tao_json()
    # gen()
    pass