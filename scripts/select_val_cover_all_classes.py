import os
import numpy as np
from PIL import Image

# Cityscapes标签目录（请根据实际路径调整）
label_dir = r'e:/deeplearning/deeplabv3plus/data/cityscapes/Cityscapes/gtFine/val'

# 递归查找所有 _labelTrainIds.png 文件
all_label_files = []
for root, dirs, files in os.walk(label_dir):
    for file in files:
        if file.endswith('_labelTrainIds.png'):
            all_label_files.append(os.path.join(root, file))

# 统计每个标签文件的 unique
class_coverage = {}
for path in all_label_files:
    img = np.array(Image.open(path))
    uniques = np.unique(img)
    class_coverage[os.path.basename(path)] = set(uniques.tolist())

# 统计每个类别被哪些图片覆盖
cover_dict = {i: [] for i in range(19)}
for fname, classes in class_coverage.items():
    for c in range(19):
        if c in classes:
            cover_dict[c].append(fname)

# 输出每个类别至少有一张图片覆盖的图片名
selected = set()
for c in range(19):
    if cover_dict[c]:
        selected.add(cover_dict[c][0])  # 每类至少选一张
    else:
        print(f'类别{c}没有任何图片覆盖！')

print('建议的验证集图片（每类至少一张）：')
for fname in sorted(selected):
    print(fname)

# 可选：输出每类覆盖的图片数量
for c in range(19):
    print(f'类别{c}被{len(cover_dict[c])}张图片覆盖')
