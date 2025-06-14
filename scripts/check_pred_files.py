import os

IMG_DIR = 'Dataset/Voc/VOCdevkit/JPEGImages'
GT_DIR = 'Dataset/Voc/VOCdevkit/SegmentationClass'
PRED_DIR = 'results/mious_deeplab/miou_out_mobilenetv2rein_2025-06-11_20-54-16_eval_2025_06_14/detection-results'

img_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
gt_files = set(f.replace('.png', '') for f in os.listdir(GT_DIR) if f.endswith('.png'))
pred_files = set(f.replace('.png', '') for f in os.listdir(PRED_DIR) if f.endswith('.png'))

missing_gt = []
missing_pred = []

for img_name in img_files:
    base = img_name.replace('.jpg', '')
    gt_name = base  # GT标签通常与原图同名，只是扩展名不同
    pred_name = base + '_gtFine_labelTrainIds'  # 预测结果通常为原图名+_gtFine_labelTrainIds
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
