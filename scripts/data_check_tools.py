"""
data_check_tools.py

本脚本整合了原 check_label_validity.py、check_pred_files.py、clean_val_txt.py、select_val_cover_all_classes.py 等数据检查与清理相关功能。
每个功能以函数形式提供，便于统一调用和维护。
如需扩展更多数据集检查、清理、统计功能，可在本脚本中补充。

合并来源：
- check_label_validity.py
- check_pred_files.py
- clean_val_txt.py
- select_val_cover_all_classes.py

用法示例：
    from data_check_tools import check_label_validity, check_pred_files, clean_val_txt, select_val_cover_all_classes
    check_label_validity(...)
    check_pred_files(...)
    clean_val_txt(...)
    select_val_cover_all_classes(...)
"""

# 以下为各原脚本功能的函数模板，具体实现可参考原脚本内容补充完善。

def check_label_validity(val_txt, label_dir, output_file=None):
    """
    检查标签文件有效性，将无效标签输出到指定文件。
    Args:
        val_txt (str): 验证集txt文件路径
        label_dir (str): 标签文件夹路径
        output_file (str|None): 检查结果输出文件（可选）
    """
    import os
    from PIL import Image
    import numpy as np
    empty_or_broken = []
    all_uniques = set()
    with open(val_txt) as f:
        names = [x.strip() for x in f.readlines()]
    for name in names:
        label_path = os.path.join(label_dir, name + '.png')
        try:
            img = np.array(Image.open(label_path))
            uniques = np.unique(img)
            if uniques.size == 0:
                empty_or_broken.append(name)
            all_uniques.update(uniques.tolist())
        except Exception as e:
            print(f'{name}: 文件损坏或无法读取 ({e})')
            empty_or_broken.append(name)
    if empty_or_broken:
        print('以下标签文件内容为空或损坏:')
        for name in empty_or_broken:
            print(name)
        if output_file:
            with open(output_file, 'w') as f:
                for name in empty_or_broken:
                    f.write(name + '\n')
    else:
        print('所有标签文件内容正常')
    print('所有标签像素值:', sorted(all_uniques))

def check_pred_files(img_dir, gt_dir, pred_dir):
    """
    检查预测结果文件的完整性和命名规范。
    Args:
        img_dir (str): 原图文件夹
        gt_dir (str): 标签文件夹
        pred_dir (str): 预测结果文件夹
    """
    import os
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    gt_files = set(f.replace('.png', '') for f in os.listdir(gt_dir) if f.endswith('.png'))
    pred_files = set(f.replace('.png', '') for f in os.listdir(pred_dir) if f.endswith('.png'))
    missing_gt = []
    missing_pred = []
    for img_name in img_files:
        base = img_name.replace('.jpg', '')
        gt_name = base
        pred_name = base + '_gtFine_labelTrainIds'
        if gt_name not in gt_files:
            missing_gt.append(gt_name + '.png')
        if pred_name not in pred_files:
            missing_pred.append(pred_name + '.png')
    extra_gt = [f + '.png' for f in gt_files if f not in set(img.replace('.jpg', '') for img in img_files)]
    extra_pred = [f + '.png' for f in pred_files if not any(f.startswith(img.replace('.jpg','')) for img in img_files)]
    print('=== 缺失的GT标签文件（与原图不对应）===')
    for name in missing_gt:
        print(name)
    if not missing_gt:
        print('无')
    print('\n=== 缺失的预测结果文件（与原图不对应）===')
    for name in missing_pred:
        print(name)
    if not missing_pred:
        print('无')
    print('\n=== 多余的GT标签文件（无对应原图）===')
    for name in extra_gt:
        print(name)
    if not extra_gt:
        print('无')
    print('\n=== 多余的预测结果文件（无对应原图）===')
    for name in extra_pred:
        print(name)
    if not extra_pred:
        print('无')

def clean_val_txt(val_txt, label_dir, output_txt):
    """
    清理验证集txt文件中的无效或重复项，只保留有标签文件的图片名。
    Args:
        val_txt (str): 验证集txt文件路径
        label_dir (str): 标签文件夹路径
        output_txt (str): 输出清理后txt路径
    """
    import os
    with open(val_txt) as f:
        names = [x.strip() for x in f.readlines()]
    with open(output_txt, 'w') as fout:
        for name in names:
            label_path = os.path.join(label_dir, name + '.png')
            if os.path.exists(label_path):
                fout.write(name + '\n')
    print(f'已生成 {output_txt}，只保留有标签文件的图片名')

def select_val_cover_all_classes(label_dir, num_classes=19):
    """
    自动选择覆盖所有类别的验证集样本，提升评估代表性。
    Args:
        label_dir (str): 标签文件夹路径
        num_classes (int): 类别数，默认19
    """
    import os
    import numpy as np
    from PIL import Image
    all_label_files = []
    for root, dirs, files in os.walk(label_dir):
        for file in files:
            if file.endswith('_labelTrainIds.png') or file.endswith('.png'):
                all_label_files.append(os.path.join(root, file))
    class_coverage = {}
    for path in all_label_files:
        img = np.array(Image.open(path))
        uniques = np.unique(img)
        class_coverage[os.path.basename(path)] = set(uniques.tolist())
    cover_dict = {i: [] for i in range(num_classes)}
    for fname, classes in class_coverage.items():
        for c in range(num_classes):
            if c in classes:
                cover_dict[c].append(fname)
    selected = set()
    for c in range(num_classes):
        if cover_dict[c]:
            selected.add(cover_dict[c][0])
        else:
            print(f'类别{c}没有任何图片覆盖！')
    print('建议的验证集图片（每类至少一张）：')
    for fname in sorted(selected):
        print(fname)
    for c in range(num_classes):
        print(f'类别{c}被{len(cover_dict[c])}张图片覆盖')

if __name__ == "__main__":
    # 可在此添加命令行参数解析，支持独立运行各功能
    pass
