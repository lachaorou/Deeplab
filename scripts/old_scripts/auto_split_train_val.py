import os
import random

label_dir = r'e:/deeplearning/deeplabv3plus/data/voc/VOCdevkit/SegmentationClass'
output_dir = r'e:/deeplearning/deeplabv3plus/data/voc/VOCdevkit/ImageSets/Segmentation'

all_names = [os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.png')]
random.shuffle(all_names)
split_idx = int(len(all_names) * 0.8)
train_names = all_names[:split_idx]
val_names = all_names[split_idx:]

with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
    for name in train_names:
        f.write(name + '\n')
with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
    for name in val_names:
        f.write(name + '\n')
print(f'已生成 train.txt({len(train_names)}) 和 val.txt({len(val_names)})，覆盖所有标签文件。')
