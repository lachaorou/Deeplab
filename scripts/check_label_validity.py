import os
from PIL import Image
import numpy as np

val_txt = r'e:\deeplearning\deeplabv3plus\data\voc\VOCdevkit\ImageSets\Segmentation\val_clean.txt'
label_dir = r'e:\deeplearning\deeplabv3plus\data\voc\VOCdevkit\VOC2012\SegmentationClass'

empty_or_broken = []
all_uniques = set()
with open(val_txt) as f:
    names = [x.strip() for x in f.readlines()]
for name in names:
    label_path = os.path.join(label_dir, name + '.png')
    try:
        img = np.array(Image.open(label_path))
        uniques = np.unique(img)
        if uniques.size == 0:
            empty_or_broken.append(name)
        all_uniques.update(uniques.tolist())
    except Exception as e:
        print(f'{name}: 文件损坏或无法读取 ({e})')
        empty_or_broken.append(name)
if empty_or_broken:
    print('以下标签文件内容为空或损坏:')
    for name in empty_or_broken:
        print(name)
else:
    print('所有标签文件内容正常')
print('所有标签像素值:', sorted(all_uniques))
