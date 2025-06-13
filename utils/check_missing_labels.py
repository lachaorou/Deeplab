import os

import os
label_dir = os.path.join('data', 'voc', 'VOCdevkit', 'SegmentationClass') if os.path.exists(os.path.join('data', 'voc', 'VOCdevkit', 'SegmentationClass')) else r'VOCdevkit/SegmentationClass'
split_txts = [
    os.path.join('data', 'voc', 'VOCdevkit', 'ImageSets', 'Segmentation', 'train.txt') if os.path.exists(os.path.join('data', 'voc', 'VOCdevkit', 'ImageSets', 'Segmentation', 'train.txt')) else r'VOCdevkit/ImageSets/Segmentation/train.txt',
    os.path.join('data', 'voc', 'VOCdevkit', 'ImageSets', 'Segmentation', 'val.txt') if os.path.exists(os.path.join('data', 'voc', 'VOCdevkit', 'ImageSets', 'Segmentation', 'val.txt')) else r'VOCdevkit/ImageSets/Segmentation/val.txt'
]

for txt_path in split_txts:
    with open(txt_path, 'r') as f:
        names = [line.strip() for line in f if line.strip()]
    missing = []
    for name in names:
        fname = name + '_gtFine_labelTrainIds.png'
        if not os.path.exists(os.path.join(label_dir, fname)):
            missing.append(fname)
    print(f'{txt_path} 缺失标签文件数: {len(missing)}')
    if missing:
        print('缺失示例:', missing[:10])
    else:
        print('全部标签文件齐全！')
