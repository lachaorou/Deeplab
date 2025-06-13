import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2

import random
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil

from models.deeplabv3_reins import DeepLab
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.colorize import colorize_mask
import torch.nn.functional as F

# 修正数据集路径
VAL_TXT = os.path.join('Dataset', 'Voc', 'VOCdevkit', 'ImageSets', 'Segmentation', 'val.txt')
IMG_DIR = os.path.join('Dataset', 'Voc', 'VOCdevkit', 'JPEGImages')
GT_DIR = os.path.join('Dataset', 'Voc', 'VOCdevkit', 'SegmentationClass')
WEIGHTS_DIR = os.path.join('results', 'logs_deeplab', 'logs_mobilenetv2rein')
OUT_DIR = os.path.join('visualize_samples')
NUM_CLASSES = 19
INPUT_SHAPE = (1024, 1536)  # (h, w)  # 升级分辨率，适配24G显存
NUM_SAMPLES = 10
BACKBONE = 'mobilenet'

# 1. 读取验证集样本
with open(VAL_TXT, 'r') as f:
    val_ids = [x.strip() for x in f.readlines() if x.strip()]

# 2. 遍历权重文件
def get_weight_files():
    files = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith('.pth') and f.startswith('ep')]
    files = sorted(files, key=lambda x: int(x.split('-')[0].replace('ep','')))
    return files

def load_model(weight_path):
    model = DeepLab(num_classes=NUM_CLASSES, backbone=BACKBONE, downsample_factor=8, pretrained=False)
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)  # 适配PyTorch安全机制
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_miou_png(model, image, input_shape, cuda=False):
    image = cvtColor(image)
    orininal_h, orininal_w = np.array(image).shape[:2]
    image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    with torch.no_grad():
        images = torch.from_numpy(image_data)
        if cuda:
            images = images.cuda()
        pr = model(images)[0]
        pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
        pr = pr[int((input_shape[0] - nh) // 2) : int((input_shape[0] - nh) // 2 + nh),
                int((input_shape[1] - nw) // 2) : int((input_shape[1] - nw) // 2 + nw)]
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)
    # 彩色可视化
    color_pred = colorize_mask(pr)
    return Image.fromarray(color_pred)

def visualize_for_checkpoint(weight_file, sample_ids):
    epoch_str = weight_file.split('-')[0]
    out_dir = os.path.join('Inspect', 'VisualizeCompare', f'val_compare_{epoch_str}')
    os.makedirs(out_dir, exist_ok=True)
    model = load_model(os.path.join(WEIGHTS_DIR, weight_file))
    rows = []
    for img_id in tqdm(sample_ids, desc=f'Visualizing {epoch_str}'):
        img_path = os.path.join(IMG_DIR, img_id + '.jpg')
        gt_path = os.path.join(GT_DIR, img_id + '.png')
        pred_path = os.path.join(out_dir, img_id + '_pred.png')
        img = Image.open(img_path)
        gt = Image.open(gt_path)
        pred = get_miou_png(model, img, INPUT_SHAPE)
        pred.save(pred_path)
        # 保存原图
        img_save_path = os.path.join(out_dir, img_id + '_img.jpg')
        img.save(img_save_path)
        # 标签灰度图
        gt_gray_save_path = os.path.join(out_dir, img_id + '_gt_gray.png')
        gt.save(gt_gray_save_path)
        # 标签彩色
        gt_color = colorize_mask(np.array(gt))
        gt_color_img = Image.fromarray(gt_color)
        gt_color_save_path = os.path.join(out_dir, img_id + '_gt_color.png')
        gt_color_img.save(gt_color_save_path)
        # 绝对路径
        abs_img = os.path.abspath(img_save_path).replace('\\', '/')
        abs_gt_color = os.path.abspath(gt_color_save_path).replace('\\', '/')
        abs_gt_gray = os.path.abspath(gt_gray_save_path).replace('\\', '/')
        abs_pred = os.path.abspath(pred_path).replace('\\', '/')
        rows.append(f'| {img_id} | ![]({abs_img}) | ![]({abs_gt_color}) | ![]({abs_gt_gray}) | ![]({abs_pred}) |')
    # 生成Markdown
    md_path = os.path.join(out_dir, 'compare.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('| ID | 原图 | 标签(彩色) | 标签(灰度) | 预测 |\n')
        f.write('|----|------|-----------|-----------|------|\n')
        for row in rows:
            f.write(row + '\n')
    print(f'可视化结果已保存到: {out_dir}')

def visualize_for_all_samples(weight_files, sample_ids):
    # 以样本为主分类
    out_root = os.path.join('Inspect', 'VisualizeCompare', 'by_sample')
    os.makedirs(out_root, exist_ok=True)
    # 预加载所有模型
    models = {}
    for weight_file in weight_files:
        epoch_str = weight_file.split('-')[0]
        model = load_model(os.path.join(WEIGHTS_DIR, weight_file))
        models[epoch_str] = model
    # 对每个样本建文件夹
    for img_id in tqdm(sample_ids, desc='By-sample visualization'):
        sample_dir = os.path.join(out_root, img_id)
        os.makedirs(sample_dir, exist_ok=True)
        img_path = os.path.join(IMG_DIR, img_id + '.jpg')
        gt_path = os.path.join(GT_DIR, img_id + '.png')
        img = Image.open(img_path)
        gt = Image.open(gt_path)
        # 保存原图
        img_save_path = os.path.join(sample_dir, img_id + '_img.jpg')
        img.save(img_save_path)
        # 标签灰度
        gt_gray_save_path = os.path.join(sample_dir, img_id + '_gt_gray.png')
        gt.save(gt_gray_save_path)
        # 标签彩色
        gt_color = colorize_mask(np.array(gt))
        gt_color_img = Image.fromarray(gt_color)
        gt_color_save_path = os.path.join(sample_dir, img_id + '_gt_color.png')
        gt_color_img.save(gt_color_save_path)
        # 预测（每个epoch一张）
        pred_paths = []
        for epoch_str, model in models.items():
            pred = get_miou_png(model, img, INPUT_SHAPE)
            pred_name = f'{epoch_str}_pred.png'
            pred_path = os.path.join(sample_dir, pred_name)
            pred.save(pred_path)
            pred_paths.append((epoch_str, pred_path))
        # 生成Markdown
        abs_img = os.path.abspath(img_save_path).replace('\\', '/')
        abs_gt_color = os.path.abspath(gt_color_save_path).replace('\\', '/')
        abs_gt_gray = os.path.abspath(gt_gray_save_path).replace('\\', '/')
        md_path = os.path.join(sample_dir, 'compare.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f'# 样本: {img_id}\n')
            f.write('| 原图 | 标签(彩色) | 标签(灰度) |')
            for epoch_str, _ in pred_paths:
                f.write(f' {epoch_str}_预测 |')
            f.write('\n|------|-----------|-----------|' + '------|'*len(pred_paths) + '\n')
            f.write(f'| ![]({abs_img}) | ![]({abs_gt_color}) | ![]({abs_gt_gray}) |')
            for _, pred_path in pred_paths:
                abs_pred = os.path.abspath(pred_path).replace('\\', '/')
                f.write(f' ![]({abs_pred}) |')
            f.write('\n')
    print(f'按样本可视化结果已保存到: {out_root}')

def main():
    weight_files = get_weight_files()
    print(f'检测到权重文件: {weight_files}')
    random.seed(42)
    sample_ids = random.sample(val_ids, min(NUM_SAMPLES, len(val_ids)))
    visualize_for_all_samples(weight_files, sample_ids)

if __name__ == '__main__':
    main()
