import os
from PIL import Image
import numpy as np

# Cityscapes官方19类配色
CITYSCAPES_COLORMAP = [
    (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153),
    (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
    ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
    (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32)
]

# detection-results 目录
input_dir = 'results/mious_deeplab/miou_out_mobilenetv2_rein/detection-results'
output_dir = input_dir + '_color'
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.endswith('.png'):
        img = Image.open(os.path.join(input_dir, fname))
        mask = np.array(img)
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(CITYSCAPES_COLORMAP):
            color_mask[mask == label] = color
        color_mask[mask == 255] = (0, 0, 0)  # ignore设为黑色
        out_img = Image.fromarray(color_mask)
        out_img.save(os.path.join(output_dir, fname))
print(f'已批量彩色化，输出目录：{output_dir}')
