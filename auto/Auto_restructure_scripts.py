import os
import shutil

# 目录结构重组脚本：将auto/、utils/、scripts/下的典型脚本按功能归类重命名到auto/standardized/
# 仅演示部分典型脚本，实际合并请人工确认

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def move_and_rename(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"[重组] {src} -> {dst}")

def restructure_scripts(project_root):
    auto_dir = os.path.join(project_root, 'auto')
    utils_dir = os.path.join(project_root, 'utils')
    scripts_dir = os.path.join(project_root, 'scripts')
    std_dir = os.path.join(auto_dir, 'standardized')
    ensure_dir(std_dir)
    # 典型脚本归类重命名
    move_and_rename(os.path.join(utils_dir, 'check_label_values.py'), os.path.join(std_dir, 'Check_LabelValues.py'))
    move_and_rename(os.path.join(utils_dir, 'stat_label_distribution.py'), os.path.join(std_dir, 'Check_LabelDistribution.py'))
    move_and_rename(os.path.join(scripts_dir, 'check_label_validity.py'), os.path.join(std_dir, 'Check_LabelValidity.py'))
    move_and_rename(os.path.join(scripts_dir, 'auto_split_train_val.py'), os.path.join(std_dir, 'Auto_SplitTrainVal.py'))
    move_and_rename(os.path.join(scripts_dir, 'visualize_saved_checkpoints.py'), os.path.join(std_dir, 'Visualize_SavedCheckpoints.py'))
    print("[完成] 典型脚本已归类重命名到 auto/standardized/，请人工检查后删除原文件。")

if __name__ == '__main__':
    ROOT = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, '..'))
    restructure_scripts(PROJECT_ROOT)
