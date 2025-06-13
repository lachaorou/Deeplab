import os
import numpy as np
from PIL import Image
from collections import Counter

# Cityscapes 19类+ignore
name_classes = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

# 自动适配 VOCdevkit 路径
import os
VOCdevkit_path = os.path.join('data', 'voc', 'VOCdevkit') if os.path.exists(os.path.join('data', 'voc', 'VOCdevkit')) else 'VOCdevkit'
seg_dir = os.path.join(VOCdevkit_path, 'SegmentationClass')

all_counts = Counter()
file_counts = {}

for fname in os.listdir(seg_dir):
    if not fname.endswith('.png'):
        continue
    mask = np.array(Image.open(os.path.join(seg_dir, fname)))
    unique, counts = np.unique(mask, return_counts=True)
    count_dict = dict(zip(unique, counts))
    file_counts[fname] = count_dict
    all_counts.update(count_dict)

print('类别像素统计:')
for i, name in enumerate(name_classes):
    print(f'{i:2d} {name:15s}: {all_counts.get(i,0)}')
print(f'255 ignore      : {all_counts.get(255,0)}')

print('\n每个标签文件中出现的类别:')
for fname, cdict in list(file_counts.items())[:10]:  # 只展示前10个文件
    print(fname, sorted(cdict.keys()))
print('...')

# 计算类别权重（像素数的倒数，归一化）
counts = np.array([all_counts.get(i, 0) for i in range(len(name_classes))], dtype=np.float64)
# 防止除零
counts[counts == 0] = 1
inv = 1.0 / counts
cls_weights = inv / inv.sum() * len(name_classes)
print('\n建议的类别权重(可直接复制到train.py):')
print('cls_weights = np.array([')
for w in cls_weights:
    print(f'    {w:.6f},')
print('], np.float32)')

# ===== 自动计算类别权重并打印，可直接复制到train.py =====
import math

cls_pixel_counts = np.array([all_counts.get(i, 0) for i in range(19)])
# 防止除零，最小像素数设为1
cls_pixel_counts = np.maximum(cls_pixel_counts, 1)
# 采用log平滑的倒数权重，常用公式：weight = 1 / log(c + p)
c = 1.02  # 平滑常数，防止log(1)=0
cls_weights = 1 / np.log(c + cls_pixel_counts)
# 归一化到均值为1
cls_weights = cls_weights / np.mean(cls_weights)

print('\n建议的类别权重cls_weights，可直接复制到train.py：')
print('cls_weights = [')
for w in cls_weights:
    print(f'    {w:.6f},')
print(']')
