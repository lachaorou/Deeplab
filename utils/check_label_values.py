import os
import random
from collections import Counter
from PIL import Image

try:
    from config import LABEL_DIR
except ImportError:
    LABEL_DIR = os.path.join('data', 'voc', 'VOCdevkit', 'SegmentationClass') if os.path.exists(os.path.join('data', 'voc', 'VOCdevkit', 'SegmentationClass')) else os.path.join('VOCdevkit', 'SegmentationClass')

SAMPLE_NUM = 10  # 随机抽查数量，可根据需要调整

all_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.png')]
if len(all_files) == 0:
    print(f'未找到标签文件，请检查路径：{LABEL_DIR}')
    exit(1)

sample_files = random.sample(all_files, min(SAMPLE_NUM, len(all_files)))
print(f'随机抽查 {len(sample_files)} 个标签文件：')

# 支持自动判断标签类型（trainId/labelId），并检查像素值是否合法
CITYSCAPES_TRAINID_SET = set(range(19)) | {255}
CITYSCAPES_LABELID_SET = set([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 255
])

# 判断标签类型
def guess_label_type(values):
    vset = set(values)
    if vset <= CITYSCAPES_TRAINID_SET:
        return 'trainId'
    elif vset <= CITYSCAPES_LABELID_SET:
        return 'labelId'
    else:
        return 'unknown'

all_values = Counter()
file_types = {}
for fname in sample_files:
    path = os.path.join(LABEL_DIR, fname)
    img = Image.open(path)
    arr = img.convert('L')
    values = list(arr.getdata())
    c = Counter(values)
    all_values.update(c)
    label_type = guess_label_type(c.keys())
    file_types[fname] = label_type
    print(f'{fname}: {sorted(c.items())}  [类型: {label_type}]')

print('\n所有抽查文件像素值分布汇总：')
for v, cnt in sorted(all_values.items()):
    print(f'像素值 {v}: {cnt} 个')

# 检查标签类型
all_type = guess_label_type(all_values.keys())
print(f'\n[标签类型自动判断] 本批标签类型为: {all_type}')
if all_type == 'unknown':
    print('[警告] 检查到非Cityscapes标准的像素值，请检查标签映射！')
    valid = CITYSCAPES_TRAINID_SET | CITYSCAPES_LABELID_SET
else:
    valid = CITYSCAPES_TRAINID_SET if all_type == 'trainId' else CITYSCAPES_LABELID_SET

abnormal = [v for v in all_values if v not in valid]
if abnormal:
    print(f'\n[警告] 检查到异常像素值: {abnormal}，请修正标签映射！')
else:
    print('\n[通过] 未发现异常像素值，标签格式基本正确。')

# 输出每个文件的类型统计，便于人工核查
print('\n各文件标签类型分布:')
for fname, t in file_types.items():
    print(f'{fname}: {t}')
