import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.colorize import colorize_mask

def auto_visualize_compare(val_lines, pred_dir, label_dir, epoch):
    """
    每5轮自动抽取10个样本进行可视化对比，保存到visualize_compare/20250606_15.31_epoch{epoch}
    """
    out_dir = f"visualize_compare/20250606_15.31_epoch{epoch}"
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

def auto_visualize_train_samples(train_lines, pred_dir, label_dir, epoch, sample_ids=None):
    """
    每5轮从训练集抽取10个样本，每个样本单独建文件夹，记录每5轮的特征可视化
    """
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
        out_dir = f"visualize_compare/train_sample_{basename}"
        os.makedirs(out_dir, exist_ok=True)
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(label_color)
        axs[0].set_title('Label')
        axs[1].imshow(pred_color)
        axs[1].set_title('Pred')
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"epoch{epoch}.png"))
        plt.close()
    return sample_ids
