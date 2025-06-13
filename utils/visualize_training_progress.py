import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.colorize import colorize_mask
from typing import List, Optional

def visualize_train_samples(train_lines: List[str], pred_dir: str, label_dir: str, epoch: int, sample_ids: Optional[List[int]] = None, out_root: str = "Inspect/VisualizeCompare") -> List[int]:
    """
    每N轮从训练集抽取10个样本，每个样本单独建文件夹，记录每N轮的分割可视化。
    返回本轮使用的sample_ids，便于后续复用。
    """
    os.makedirs(out_root, exist_ok=True)
    if sample_ids is None:
        np.random.seed(42)
        sample_ids = np.random.choice(len(train_lines), min(10, len(train_lines)), replace=False)
        sample_ids = list(sample_ids)
    for idx in sample_ids:
        basename = train_lines[idx].strip()
        pred_path = os.path.join(pred_dir, basename + ".png")
        label_path = os.path.join(label_dir, basename + ".png")
        if not (os.path.exists(pred_path) and os.path.exists(label_path)):
            continue
        pred = np.array(Image.open(pred_path))
        label = np.array(Image.open(label_path))
        pred_color = colorize_mask(pred)
        label_color = colorize_mask(label)
        sample_dir = os.path.join(out_root, f"train_sample_{basename}")
        os.makedirs(sample_dir, exist_ok=True)
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(label_color)
        axs[0].set_title('Label')
        axs[1].imshow(pred_color)
        axs[1].set_title('Pred')
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, f"epoch{epoch}.png"))
        plt.close()
    return sample_ids

def visualize_val_compare(val_lines: List[str], pred_dir: str, label_dir: str, epoch: int, out_root: str = "Inspect/VisualizeCompare"):
    """
    每N轮自动抽取10个验证样本进行可视化对比，按epoch归档。
    """
    out_dir = os.path.join(out_root, f"val_compare_epoch{epoch}")
    os.makedirs(out_dir, exist_ok=True)
    import random
    sample_lines = random.sample(val_lines, min(10, len(val_lines)))
    for line in sample_lines:
        basename = line.strip()
        pred_path = os.path.join(pred_dir, basename + ".png")
        label_path = os.path.join(label_dir, basename + ".png")
        if not (os.path.exists(pred_path) and os.path.exists(label_path)):
            continue
        pred = np.array(Image.open(pred_path))
        label = np.array(Image.open(label_path))
        pred_color = colorize_mask(pred)
        label_color = colorize_mask(label)
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(label_color)
        axs[0].set_title('Label')
        axs[1].imshow(pred_color)
        axs[1].set_title('Pred')
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{basename}.png"))
        plt.close()

def update_markdown_table(sample_dir: str, out_md: str = None):
    """
    自动生成/更新Markdown对比表格，按样本文件夹下所有epoch图片生成对比表。
    sample_dir: 某个train_sample_xxx文件夹路径
    out_md: 输出的Markdown文件路径，默认在sample_dir下
    """
    if out_md is None:
        out_md = os.path.join(sample_dir, "compare.md")
    images = sorted([f for f in os.listdir(sample_dir) if f.endswith('.png')])
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(f"# 训练过程分割对比 ({os.path.basename(sample_dir)})\n\n")
        f.write("| Epoch | 分割结果 |\n")
        f.write("|-------|----------|\n")
        for img in images:
            epoch = img.replace("epoch", "").replace(".png", "")
            f.write(f"| {epoch} | ![]({img}) |\n")
    return out_md

def batch_update_all_markdown(out_root: str = "Inspect/VisualizeCompare"):
    """
    批量为所有train_sample_xxx文件夹生成/更新compare.md
    """
    for d in os.listdir(out_root):
        sample_dir = os.path.join(out_root, d)
        if os.path.isdir(sample_dir) and d.startswith("train_sample_"):
            update_markdown_table(sample_dir)
