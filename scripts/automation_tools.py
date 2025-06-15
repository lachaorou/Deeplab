"""
automation_tools.py

本脚本整合了原 auto_colorize_detection_results.py、auto_resume_train.py、auto_split_train_val.py 等自动化/批量辅助相关功能。
每个功能以函数形式提供，便于统一调用和维护。
如需扩展更多自动化脚本功能，可在本脚本中补充。

合并来源：
- auto_colorize_detection_results.py
- auto_resume_train.py
- auto_split_train_val.py

用法示例：
    from automation_tools import auto_colorize_results, auto_resume_train, auto_split_train_val
    auto_colorize_results(...)
    auto_resume_train(...)
    auto_split_train_val(...)
"""

def auto_colorize_results(input_dir, output_dir=None):
    """
    批量上色检测结果，便于可视化分析。
    """
    import os
    from PIL import Image
    import numpy as np
    CITYSCAPES_COLORMAP = [
        (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153),
        (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
        ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
        (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32)
    ]
    if output_dir is None:
        output_dir = input_dir + '_color'
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.endswith('.png'):
            img = Image.open(os.path.join(input_dir, fname))
            mask = np.array(img)
            color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(CITYSCAPES_COLORMAP):
                color_mask[mask == label] = color
            color_mask[mask == 255] = (0, 0, 0)
            out_img = Image.fromarray(color_mask)
            out_img.save(os.path.join(output_dir, fname))
    print(f'已批量彩色化，输出目录：{output_dir}')

def auto_resume_train(log_dir, total_epoch=100, num_classes=19, backbone='xception', pretrained=True):
    """
    自动断点续训，检测最新权重恢复训练。
    """
    import os
    import re
    import subprocess
    def find_latest_checkpoint(log_dir):
        if not os.path.exists(log_dir):
            print(f"日志目录不存在: {log_dir}")
            return None, None
        files = os.listdir(log_dir)
        epoch_files = [f for f in files if re.match(r'epoch_(\\d+).pth', f)]
        if epoch_files:
            epochs = [int(re.findall(r'epoch_(\\d+).pth', f)[0]) for f in epoch_files]
            max_idx = epochs.index(max(epochs))
            return os.path.join(log_dir, epoch_files[max_idx]), max(epochs)
        if 'best_epoch_weights.pth' in files:
            return os.path.join(log_dir, 'best_epoch_weights.pth'), None
        return None, None
    ckpt_path, last_epoch = find_latest_checkpoint(log_dir)
    if ckpt_path is None:
        print('未找到已保存的权重，将从头训练...')
        cmd = f"python train.py --num_classes {num_classes} --backbone {backbone} --pretrained {pretrained} --epoch {total_epoch}"
    else:
        print(f'检测到最近权重: {ckpt_path}')
        if last_epoch is not None:
            left_epoch = total_epoch - last_epoch
            if left_epoch <= 0:
                print('训练已完成，无需续训。')
                return
            cmd = f"python train.py --num_classes {num_classes} --backbone {backbone} --pretrained {pretrained} --init_epoch {last_epoch} --epoch {total_epoch}"
        else:
            print('检测到best_epoch_weights.pth，建议手动指定epoch。')
            cmd = f"python train.py --num_classes {num_classes} --backbone {backbone} --pretrained {pretrained} --epoch {total_epoch} --model_path {ckpt_path}"
    print(f'自动续训命令：{cmd}')
    subprocess.call(cmd, shell=True)

def auto_split_train_val(label_dir, output_dir, ratio=0.8):
    """
    自动划分训练/验证集，支持自定义比例。
    """
    import os
    import random
    all_names = [os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.png')]
    random.shuffle(all_names)
    split_idx = int(len(all_names) * ratio)
    train_names = all_names[:split_idx]
    val_names = all_names[split_idx:]
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for name in train_names:
            f.write(name + '\n')
    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        for name in val_names:
            f.write(name + '\n')
    print(f'已生成 train.txt({len(train_names)}) 和 val.txt({len(val_names)})，覆盖所有标签文件。')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeeplabV3+ 自动化工具集")
    subparsers = parser.add_subparsers(dest='command')

    parser_color = subparsers.add_parser('colorize', help='批量彩色化检测结果')
    parser_color.add_argument('--input_dir', type=str, required=True)
    parser_color.add_argument('--output_dir', type=str, default=None)

    parser_resume = subparsers.add_parser('resume', help='自动断点续训')
    parser_resume.add_argument('--log_dir', type=str, required=True)
    parser_resume.add_argument('--total_epoch', type=int, default=100)
    parser_resume.add_argument('--num_classes', type=int, default=19)
    parser_resume.add_argument('--backbone', type=str, default='xception')
    parser_resume.add_argument('--pretrained', type=bool, default=True)

    parser_split = subparsers.add_parser('split', help='自动划分训练/验证集')
    parser_split.add_argument('--label_dir', type=str, required=True)
    parser_split.add_argument('--output_dir', type=str, required=True)
    parser_split.add_argument('--ratio', type=float, default=0.8)

    args = parser.parse_args()
    if args.command == 'colorize':
        auto_colorize_results(args.input_dir, args.output_dir)
    elif args.command == 'resume':
        auto_resume_train(args.log_dir, args.total_epoch, args.num_classes, args.backbone, args.pretrained)
    elif args.command == 'split':
        auto_split_train_val(args.label_dir, args.output_dir, args.ratio)
    else:
        parser.print_help()
