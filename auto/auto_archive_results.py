import os
import shutil
from datetime import datetime

"""
自动整理 logs 和 miou 相关文件/目录到 results/logs_deeplab 和 results/mious_deeplab。
- logs_xxx/、loss_*/ 归档到 results/logs_deeplab
- miou_out*/、detection-results/ 归档到 results/mious_deeplab
- 支持按日期自动归档，防止覆盖

用法：
1. 运行本脚本即可自动整理归档。
"""

# ====== 配置部分 ======
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_dir = os.path.join(project_root, 'results')
logs_dir = os.path.join(results_dir, 'logs_deeplab')
mious_dir = os.path.join(results_dir, 'mious_deeplab')
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(mious_dir, exist_ok=True)

# 需要归档的目录/文件名关键字
log_keywords = ['logs_', 'loss_']
miou_keywords = ['miou_out', 'detection-results']

# ====== 自动整理归档 ======
now = datetime.now().strftime('%Y%m%d_%H%M%S')
logs_subdir = os.path.join(logs_dir, f'archive_{now}')
mious_subdir = os.path.join(mious_dir, f'archive_{now}')
os.makedirs(logs_subdir, exist_ok=True)
os.makedirs(mious_subdir, exist_ok=True)

for name in os.listdir(project_root):
    for kw in log_keywords:
        if name.startswith(kw) or name == kw:
            src = os.path.join(project_root, name)
            dst = os.path.join(logs_subdir, name)
            if os.path.isdir(src):
                print(f'归档日志目录: {src} -> {dst}')
                shutil.copytree(src, dst)
            elif os.path.isfile(src):
                print(f'归档日志文件: {src} -> {dst}')
                shutil.copy2(src, dst)
    for kw in miou_keywords:
        if name.startswith(kw) or name == kw:
            src = os.path.join(project_root, name)
            dst = os.path.join(mious_subdir, name)
            if os.path.isdir(src):
                print(f'归档miou目录: {src} -> {dst}')
                shutil.copytree(src, dst)
            elif os.path.isfile(src):
                print(f'归档miou文件: {src} -> {dst}')
                shutil.copy2(src, dst)

print(f'已归档所有 logs/miou 相关文件到 {logs_subdir} 和 {mious_subdir}')
