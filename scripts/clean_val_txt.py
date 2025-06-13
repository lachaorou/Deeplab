import os

val_txt = r'e:\deeplearning\deeplabv3plus\data\voc\VOCdevkit\ImageSets\Segmentation\val.txt'
label_dir = r'e:\deeplearning\deeplabv3plus\data\voc\VOCdevkit\VOC2012\SegmentationClass'
output_txt = r'e:\deeplearning\deeplabv3plus\data\voc\VOCdevkit\ImageSets\Segmentation\val_clean.txt'

with open(val_txt) as f:
    names = [x.strip() for x in f.readlines()]
with open(output_txt, 'w') as fout:
    for name in names:
        label_path = os.path.join(label_dir, name + '.png')
        if os.path.exists(label_path):
            fout.write(name + '\n')
print('已生成 val_clean.txt，只保留有标签文件的图片名')
