import os
import json
from collections import defaultdict
from tqdm import tqdm


def gen():
    json_path = 'datasets/tao/annotations/train.json'
    img_prefix = 'datasets/tao/frames'
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    all_ids = set([i for i in range(1, 2000)])  # 2000 is larger than the max category id in TAO-OW.
    # `knowns` includes 78 TAO_category_ids that corresponds to 78 COCO classes.
    # (The other 2 COCO classes do not have corresponding classes in TAO).
    knowns = {4, 13, 1038, 544, 1057, 34, 35, 36, 41, 45, 58, 60, 579, 1091, 1097, 1099, 78, 79, 81, 91, 1115,
                    1117, 95, 1122, 99, 1132, 621, 1135, 625, 118, 1144, 126, 642, 1155, 133, 1162, 139, 154, 174, 185,
                    699, 1215, 714, 717, 1229, 211, 729, 221, 229, 747, 235, 237, 779, 276, 805, 299, 829, 852, 347,
                    371, 382, 896, 392, 926, 937, 428, 429, 961, 452, 979, 980, 982, 475, 480, 993, 1001, 502, 1018}
    # `distractors` is defined as in the paper "Opening up Open-World Tracking"
    distractors = {20, 63, 108, 180, 188, 204, 212, 247, 303, 403, 407, 415, 490, 504, 507, 513, 529, 567,
                        569, 588, 672, 691, 702, 708, 711, 720, 736, 737, 798, 813, 815, 827, 831, 851, 877, 883,
                        912, 971, 976, 1130, 1133, 1134, 1169, 1184, 1220}
    unknowns = all_ids.difference(knowns.union(distractors))

    annos = data['annotations']
    annos = [anno for anno in annos if anno['category_id'] in knowns]
    vids = list(set([anno['video_id'] for anno in annos]))
    imgidmap, vidmap = dict(), dict()
    for vinfo in data['videos']:
        vidmap[vinfo['id']] = vinfo
    for imginfo in data['images']:
        imgidmap[imginfo['id']] = imginfo

    # 补一下anno 的frame_index
    for anno in annos:
        img_id = anno['image_id']
        anno['frame_index'] = imgidmap[img_id]['frame_index']

    total_anno_id = max([anno['id'] for anno in data['annotations']]) + 1
    total_img_id = max([imginfo['id'] for imginfo in data['images']]) + 1
    new_annos = []
    new_imgs = []
    for vid in vids:  # 逐视频 避免track_id是不是全局唯一
        vinfo = vidmap[vid]
        video_dir = os.path.join(img_prefix, vinfo['name'])
        frames = os.listdir(video_dir)
        frames = sorted(frames)
        frames = [os.path.join(vinfo['name'], fname) for fname in frames]
        
        # 检查一下对应的frame_index是否正确  不能正确对应 所以直接添加一个新的
        fname2index = {}
        for idx, fname in enumerate(frames):
            fname2index[fname] = idx
        
        track2anno = defaultdict(list)
        for anno in annos:
            if anno['video_id'] != vid:
                continue
            fname = imgidmap[anno['image_id']]['file_name']
            anno['new_index'] = fname2index[fname]
            track_id = anno['track_id']
            track2anno[track_id].append(anno)
        for k,v in track2anno.items():
            track2anno[k] = sorted(v, key=lambda x: x['frame_index'])
        
        # 逐轨迹
        for track_id, annos in tqdm(track2anno.items()):
            # print('{}/{}'.format())
            if len(annos) == 1:
                pass  # [TODO] 直接添加不补充
            else:
                for i in range(len(annos)-1):
                    anno = annos[i]
                    nanno = annos[i + 1]
                    sx1, sy1, sw, sh = anno['bbox']
                    sx2, sy2 = sx1 + sw, sy1 + sh
                    
                    ex1, ey1, ew, eh = nanno['bbox']
                    ex2, ey2 = ex1 + ew, ey1 + eh
                    gap = nanno['new_index'] - anno['new_index']
                    assert gap > 0
                    for fidx in range(anno['new_index']+1, nanno['new_index']):
                        x1 = sx1 * (fidx - anno['new_index']) / gap + ex1 * (nanno['new_index'] - fidx) / gap
                        y1 = sy1 * (fidx - anno['new_index']) / gap + ey1 * (nanno['new_index'] - fidx) / gap
                        x2 = sx2 * (fidx - anno['new_index']) / gap + ex2 * (nanno['new_index'] - fidx) / gap
                        y2 = sy2 * (fidx - anno['new_index']) / gap + ey2 * (nanno['new_index'] - fidx) / gap
                        w = x2 - x1
                        h = y2 - x2
                        new_annos.append(
                            {
                                "segmentation": None,  # [TODO]
                                "bbox": [x1, y1, w, h],
                                "area": w * h,
                                "iscrowd": anno['iscrowd'],
                                "id": total_anno_id,
                                "category_id": anno['category_id'],
                                "track_id": anno['track_id'],
                                "scale_category": anno['scale_category'],
                                "video_id": anno['video_id'],
                                "image_id": anno['image_id'],
                                "gen": 1  # 表示生成
                            }
                        )
                        total_anno_id += 1
                        new_imgs.append(
                            {
                                'id': total_img_id,
                                'video': imgidmap[anno['image_id']]['video'],
                                'width': imgidmap[anno['image_id']]['width'],
                                'height': imgidmap[anno['image_id']]['height'],
                                'file_name': frames[fidx],
                                'frame_index': fidx - anno['new_index'] + anno['frame_index'],
                                'new_index': fidx,
                                'video_id': imgidmap[anno['image_id']]['video_id'],
                                'gen': 1
                            }
                        )
                        total_img_id += 1
    # 将新生成的标签加入原本的文件
    data['annotations'].extend(new_annos)
    data['images'].extend(new_imgs)
    save_path = 'dataset_extra/train.json'
    with open(save_path, 'w') as f:
        f.write(json.dumps(data))
    
if __name__ == '__main__':
    gen()
    pass