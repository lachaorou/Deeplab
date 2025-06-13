import os
import shutil
from datetime import datetime

"""
自动整理 logs 和 miou 相关文件/目录，归档到 results/logs_deeplab 和 results/mious_deeplab。
- logs_*/loss_*/detection-results/ 归档到 results/logs_deeplab/下（带时间戳子文件夹）。
- miou_out*/miou_out_*/ 归档到 results/mious_deeplab/下（带时间戳子文件夹）。
- 归档后输出归档路径。

用法：
1. 直接运行本脚本。
"""

# ====== 配置部分 ======
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_dir = os.path.join(project_root, 'results')
logs_dir = os.path.join(results_dir, 'logs_deeplab')
mious_dir = os.path.join(results_dir, 'mious_deeplab')
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(mious_dir, exist_ok=True)

# 需要归档的目录/文件名关键字
log_keywords = ['logs_', 'loss_', 'detection-results']
miou_keywords = ['miou_out']

# ====== 自动整理归档 ======
now = datetime.now().strftime('%Y%m%d_%H%M%S')
logs_subdir = os.path.join(logs_dir, f'archive_{now}')
mious_subdir = os.path.join(mious_dir, f'archive_{now}')
os.makedirs(logs_subdir, exist_ok=True)
os.makedirs(mious_subdir, exist_ok=True)

for name in os.listdir(project_root):
    src = os.path.join(project_root, name)
    # 归档 logs/loss/detection-results
    for kw in log_keywords:
        if name.startswith(kw) or name == kw:
            dst = os.path.join(logs_subdir, name)
            if os.path.isdir(src):
                print(f'归档目录: {src} -> {dst}')
                shutil.copytree(src, dst)
            elif os.path.isfile(src):
                print(f'归档文件: {src} -> {dst}')
                shutil.copy2(src, dst)
    # 归档 miou_out 相关
    for kw in miou_keywords:
        if name.startswith(kw) or name == kw:
            dst = os.path.join(mious_subdir, name)
            if os.path.isdir(src):
                print(f'归档目录: {src} -> {dst}')
                shutil.copytree(src, dst)
            elif os.path.isfile(src):
                print(f'归档文件: {src} -> {dst}')
                shutil.copy2(src, dst)

print(f'已归档 logs/相关到 {logs_subdir}')
print(f'已归档 miou/相关到 {mious_subdir}')
