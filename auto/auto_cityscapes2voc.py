import os
import random
from glob import glob

"""
自动批量生成VOC格式train.txt/val.txt，并支持Cityscapes等数据集批量转换为VOC格式。
- 支持从Cityscapes leftImg8bit/gtFine目录批量拷贝图片/标签到VOC格式目录（JPEGImages/SegmentationClass）。
- 自动生成train.txt/val.txt，主名一一对应。
- 可自定义划分比例。

用法：
1. 配置cityscapes_root、voc_root路径。
2. 运行本脚本即可自动完成转换和txt生成。
"""

# ====== 配置部分 ======
cityscapes_root = r"data/cityscapes/Cityscapes"  # Cityscapes原始根目录
voc_root = r"data/voc/VOCdevkit"                 # VOC格式根目录
train_ratio = 0.8                                 # 训练集比例
random_seed = 42

# ====== 1. 批量拷贝图片和标签到VOC格式目录 ======
leftImg_dirs = [
    os.path.join(cityscapes_root, "leftImg8bit", split)
    for split in ["train", "val"]
]
gtFine_dirs = [
    os.path.join(cityscapes_root, "gtFine", split)
    for split in ["train", "val"]
]

jpeg_dir = os.path.join(voc_root, "JPEGImages")
seg_dir = os.path.join(voc_root, "SegmentationClass")
os.makedirs(jpeg_dir, exist_ok=True)
os.makedirs(seg_dir, exist_ok=True)

# 拷贝图片和标签
for split, left_dir, gt_dir in zip(["train", "val"], leftImg_dirs, gtFine_dirs):
    for city in os.listdir(left_dir):
        img_files = glob(os.path.join(left_dir, city, "*_leftImg8bit.png"))
        for img_path in img_files:
            basename = os.path.basename(img_path).replace("_leftImg8bit.png", "")
            voc_img = os.path.join(jpeg_dir, basename + ".jpg")
            # 转jpg
            if not os.path.exists(voc_img):
                from PIL import Image
                img = Image.open(img_path)
                img.convert("RGB").save(voc_img, quality=95)
        label_files = glob(os.path.join(gt_dir, city, "*_gtFine_labelTrainIds.png"))
        for label_path in label_files:
            basename = os.path.basename(label_path).replace("_gtFine_labelTrainIds.png", "")
            voc_label = os.path.join(seg_dir, basename + ".png")
            if not os.path.exists(voc_label):
                from PIL import Image
                label = Image.open(label_path)
                label.save(voc_label)

# ====== 2. 自动生成train.txt/val.txt ======
img_names = set([os.path.splitext(f)[0] for f in os.listdir(jpeg_dir) if f.endswith('.jpg')])
label_names = set([os.path.splitext(f)[0] for f in os.listdir(seg_dir) if f.endswith('.png')])
common_names = sorted(list(img_names & label_names))

random.seed(random_seed)
random.shuffle(common_names)
split_idx = int(len(common_names) * train_ratio)
train_names = common_names[:split_idx]
val_names = common_names[split_idx:]

output_train = os.path.join(voc_root, "ImageSets/Segmentation/train.txt")
output_val = os.path.join(voc_root, "ImageSets/Segmentation/val.txt")
os.makedirs(os.path.dirname(output_train), exist_ok=True)

with open(output_train, 'w') as f:
    for name in train_names:
        f.write(name + '\n')
with open(output_val, 'w') as f:
    for name in val_names:
        f.write(name + '\n')

print(f"已生成 train.txt: {len(train_names)} 张，val.txt: {len(val_names)} 张")
print(f"图片/标签已批量转换为VOC格式，保存在 {jpeg_dir} 和 {seg_dir}")
conda activate deeplab
python auto/auto_cityscapes2voc.py