import os
import re

# 脚本批量重命名工具：将auto/、utils/、scripts/下的脚本按统一命名规范批量重命名
# 规范示例：Auto_、Archive_、Augment_、Check_、Visualize_等

def batch_rename_scripts(target_dirs, rename_map):
    for dir_path in target_dirs:
        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)
            if not os.path.isfile(fpath):
                continue
            for pat, repl in rename_map.items():
                if re.match(pat, fname, re.IGNORECASE):
                    new_name = re.sub(pat, repl, fname, flags=re.IGNORECASE)
                    new_path = os.path.join(dir_path, new_name)
                    os.rename(fpath, new_path)
                    print(f"[重命名] {fname} -> {new_name}")
                    break
    print("[完成] 批量重命名。")

if __name__ == '__main__':
    ROOT = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, '..'))
    auto_dir = os.path.join(PROJECT_ROOT, 'auto')
    utils_dir = os.path.join(PROJECT_ROOT, 'utils')
    scripts_dir = os.path.join(PROJECT_ROOT, 'scripts')
    # 典型命名规则
    rename_map = {
        r'^a_': 'Auto_',
        r'^ar_': 'Archive_',
        r'^aug_': 'Augment_',
        r'^c_': 'Check_',
        r'^vi_': 'Visualize_',
    }
    batch_rename_scripts([auto_dir, utils_dir, scripts_dir], rename_map)
