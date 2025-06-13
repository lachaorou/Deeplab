import os
import shutil

# 自动清理项目中的__pycache__、无用txt、历史压缩包等
# 并可选归档到backup/目录

def clean_pycache(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == '__pycache__':
                full_path = os.path.join(dirpath, dirname)
                print(f"[清理] 删除: {full_path}")
                shutil.rmtree(full_path, ignore_errors=True)

def clean_tmp_and_zip(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pyc') or filename.endswith('.pyo') or filename.endswith('.tmp') or filename.endswith('.7z') or filename.endswith('.zip'):
                full_path = os.path.join(dirpath, filename)
                print(f"[清理] 删除: {full_path}")
                os.remove(full_path)
        for filename in filenames:
            if filename.lower().startswith('readme') and filename.endswith('.txt'):
                full_path = os.path.join(dirpath, filename)
                print(f"[清理] 删除: {full_path}")
                os.remove(full_path)

def backup_project(root_dir, backup_dir='backup'):
    import datetime
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f'backup_{dt}.zip'
    backup_path = os.path.join(backup_dir, backup_name)
    print(f"[备份] 打包: {backup_path}")
    shutil.make_archive(backup_path.replace('.zip',''), 'zip', root_dir)

if __name__ == '__main__':
    ROOT = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, '..'))
    backup_project(PROJECT_ROOT)
    clean_pycache(PROJECT_ROOT)
    clean_tmp_and_zip(PROJECT_ROOT)
    print("[完成] 自动清理与备份已执行。")
