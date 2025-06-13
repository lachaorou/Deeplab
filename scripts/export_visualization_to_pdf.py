import os
import img2pdf

# 配置
VIS_BASE = r"e:/baozi/deeplabv3plus/Inspect/VisualizeCompare"
OUT_SAMPLE_PDF = r"e:/baozi/deeplabv3plus/Inspect/VisualizeCompare/by_sample_visualization.pdf"
OUT_EPOCH_PDF = r"e:/baozi/deeplabv3plus/Inspect/VisualizeCompare/by_epoch_visualization.pdf"

# 1. 按样本为主
sample_root = os.path.join(VIS_BASE, "by_sample")
sample_names = [d for d in os.listdir(sample_root) if os.path.isdir(os.path.join(sample_root, d))]
sample_img_list = []
for sample in sample_names:
    folder = os.path.join(sample_root, sample)
    img_files = []
    for f in os.listdir(folder):
        if f.endswith("_img.jpg") or f.endswith("_gt_color.png") or f.endswith("_gt_gray.png") or (f.startswith("ep") and f.endswith("_pred.png")):
            img_files.append(os.path.abspath(os.path.join(folder, f)))
    img_files = sorted(img_files)
    sample_img_list.extend(img_files)
if sample_img_list:
    with open(OUT_SAMPLE_PDF, "wb") as f:
        f.write(img2pdf.convert(sample_img_list))
    print(f"已生成按样本可视化PDF: {OUT_SAMPLE_PDF}")

# 2. 按轮次为主
epoch_folders = [d for d in os.listdir(VIS_BASE) if d.startswith("val_compare_epoch")]
epoch_folders = sorted(epoch_folders, key=lambda x: int(x.split("epoch")[-1]))
epoch_img_list = []
for epf in epoch_folders:
    folder = os.path.join(VIS_BASE, epf)
    for f in os.listdir(folder):
        if f.endswith(".png"):
            epoch_img_list.append(os.path.abspath(os.path.join(folder, f)))
if epoch_img_list:
    with open(OUT_EPOCH_PDF, "wb") as f:
        f.write(img2pdf.convert(epoch_img_list))
    print(f"已生成按轮次可视化PDF: {OUT_EPOCH_PDF}")
