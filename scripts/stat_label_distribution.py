import os
import numpy as np
from PIL import Image
from collections import Counter

# 配置
LABEL_DIR = 'Dataset/Voc/VOCdevkit/SegmentationClass'
TRAIN_LIST = 'Dataset/Voc/VOCdevkit/ImageSets/Segmentation/train.txt'
NUM_CLASSES = 19  # Cityscapes/VOC 19类

# 读取训练集图片名
with open(TRAIN_LIST, 'r') as f:
    img_names = [line.strip() for line in f.readlines()]

class_counts = Counter()
total_pixels = np.int64(0)

for name in img_names:
    label_path = os.path.join(LABEL_DIR, name + '.png')
    if not os.path.exists(label_path):
        print(f'标签缺失: {label_path}')
        continue
    label = np.array(Image.open(label_path))
    mask = (label != 255)  # 忽略255区域
    total_pixels += np.sum(mask)  # 只在每张图片累加一次
    for c in range(NUM_CLASSES):
        count = np.sum((label == c))
        class_counts[c] += count

print('类别	像素数	占比')
for c in range(NUM_CLASSES):
    count = class_counts[c]
    ratio = count / total_pixels if total_pixels > 0 else 0
    print(f'{c}\t{count}\t{ratio:.4%}')

print(f'总像素数: {total_pixels}')
print('建议：小类别可考虑loss加权（如类别权重=1/占比），或采样时适当过采样小类别样本。')
