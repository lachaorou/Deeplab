import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import argparse
import subprocess

def find_latest_checkpoint(log_dir):
    """
    查找log_dir下最新的epoch权重文件和其epoch编号。
    支持epoch_XX.pth和best_epoch_weights.pth。
    """
    if not os.path.exists(log_dir):
        print(f"日志目录不存在: {log_dir}")
        return None, None
    files = os.listdir(log_dir)
    epoch_files = [f for f in files if re.match(r'epoch_(\\d+).pth', f)]
    if epoch_files:
        # 取最大epoch号
        epochs = [int(re.findall(r'epoch_(\\d+).pth', f)[0]) for f in epoch_files]
        max_idx = epochs.index(max(epochs))
        return os.path.join(log_dir, epoch_files[max_idx]), max(epochs)
    # 兜底：找best_epoch_weights.pth
    if 'best_epoch_weights.pth' in files:
        return os.path.join(log_dir, 'best_epoch_weights.pth'), None
    return None, None

def main():
    parser = argparse.ArgumentParser(description='自动断点续训脚本')
    parser.add_argument('--log_dir', type=str, default='logs_xception_rein', help='权重日志目录')
    parser.add_argument('--total_epoch', type=int, default=100, help='目标总训练轮数')
    parser.add_argument('--num_classes', type=int, default=19, help='类别数')
    parser.add_argument('--backbone', type=str, default='xception', help='主干网络')
    parser.add_argument('--pretrained', type=bool, default=True, help='是否加载主干预训练')
    args = parser.parse_args()

    ckpt_path, last_epoch = find_latest_checkpoint(args.log_dir)
    if ckpt_path is None:
        print('未找到已保存的权重，将从头训练...')
        cmd = f"python train.py --num_classes {args.num_classes} --backbone {args.backbone} --pretrained {args.pretrained} --epoch {args.total_epoch}"
    else:
        print(f'检测到最近权重: {ckpt_path}')
        if last_epoch is not None:
            left_epoch = args.total_epoch - last_epoch
            if left_epoch <= 0:
                print('训练已完成，无需续训。')
                return
            # 自动设置init_epoch参数，保证断点续训起点正确
            cmd = f"python train.py --num_classes {args.num_classes} --backbone {args.backbone} --pretrained {args.pretrained} --init_epoch {last_epoch} --epoch {args.total_epoch}"
        else:
            # best_epoch_weights.pth，无法推断epoch，建议人工指定
            print('检测到best_epoch_weights.pth，建议手动指定epoch。')
            cmd = f"python train.py --num_classes {args.num_classes} --backbone {args.backbone} --pretrained {args.pretrained} --epoch {args.total_epoch} --model_path {ckpt_path}"
    print(f'自动续训命令：{cmd}')
    subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    main()
