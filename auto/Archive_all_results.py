import os
import shutil
import datetime

# 自动归档 results/ 下所有实验结果到 results/archive_时间戳 目录

def archive_all_results(results_dir='results'):
    dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_dir = os.path.join(results_dir, f'archive_{dt}')
    os.makedirs(archive_dir, exist_ok=True)
    for name in os.listdir(results_dir):
        if name.startswith('archive_') or name == 'README.md':
            continue
        src = os.path.join(results_dir, name)
        dst = os.path.join(archive_dir, name)
        shutil.move(src, dst)
        print(f"[归档] {name} -> {archive_dir}/{name}")
    print("[完成] 所有实验结果已归档。")

if __name__ == '__main__':
    ROOT = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, '..'))
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    archive_all_results(results_dir)
