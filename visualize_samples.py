import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from cityscapesscripts.helpers.labels import trainId2label

# 配置
SAMPLES = 5  # 可调整
IMG_DIR = 'Dataset/Voc/VOCdevkit/JPEGImages'  # 原图目录
GT_DIR = 'Dataset/Voc/VOCdevkit/SegmentationClass'  # GT标签目录
PRED_DIR = 'results/mious_deeplab/miou_out_mobilenetv2rein_2025-06-11_20-54-16_eval_2025_06_14/detection-results'  # 预测结果目录
OUT_DIR = 'results/vis_samples'  # 输出目录
os.makedirs(OUT_DIR, exist_ok=True)

# 获取官方调色板
palette = np.zeros((256, 3), dtype=np.uint8)
for label in trainId2label:
    if label >= 0 and label < 19:
        palette[label] = trainId2label[label].color

def colorize_mask(mask):
    return palette[mask]

def vis_sample(img_name):
    img = Image.open(os.path.join(IMG_DIR, img_name)).convert('RGB')
    gt = Image.open(os.path.join(GT_DIR, img_name.replace('.jpg', '.png')))
    # 修正预测文件名适配逻辑，避免重复_gtFine_labelTrainIds
    base = img_name.replace('.jpg', '')
    if base.endswith('_gtFine_labelTrainIds'):
        base = base[:-len('_gtFine_labelTrainIds')]
    pred_name = base + '_gtFine_labelTrainIds.png'
    pred_path = os.path.join(PRED_DIR, pred_name)
    pred = Image.open(pred_path)
    gt_color = Image.fromarray(colorize_mask(np.array(gt)))
    pred_color = Image.fromarray(colorize_mask(np.array(pred)))
    # 拼接
    vis = Image.new('RGB', (img.width * 3, img.height))
    vis.paste(img, (0, 0))
    vis.paste(gt_color, (img.width, 0))
    vis.paste(pred_color, (img.width * 2, 0))
    vis.save(os.path.join(OUT_DIR, img_name))

if __name__ == '__main__':
    img_list = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    samples = random.sample(img_list, SAMPLES)
    for img_name in samples:
        vis_sample(img_name)
    print(f'可视化已保存到 {OUT_DIR}')
