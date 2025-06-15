"""
批量运行Reins消融实验脚本
- 支持single、multi、no_reins三种模式
- 每种模式可自定义参数组合
- 自动归档config.txt到对应实验目录
- 需在conda环境deeplab下运行
"""
import os
import subprocess
from datetime import datetime

# 配置消融实验参数
EXPERIMENTS = [
    {
        'mode': 'single',
        'params': [
            {'reins_pos': 'aspp', 'reins_channels': 64},
            {'reins_pos': 'backbone', 'reins_channels': 32},
        ]
    },
    {
        'mode': 'multi',
        'params': [
            {'reins_pos': 'aspp+backbone', 'reins_channels': 32},
        ]
    },
    {
        'mode': 'no_reins',
        'params': [
            {'reins_pos': 'none', 'reins_channels': 0},
        ]
    }
]

TRAIN_SCRIPT = os.path.join('scripts', 'train.py')
RESULTS_DIR = os.path.join('results', 'reins_ablation')


def run_experiment(mode, param_dict):
    # 构建实验目录
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{mode}_reins_{param_dict['reins_pos']}_{param_dict['reins_channels']}_{time_str}"
    exp_dir = os.path.join(RESULTS_DIR, mode, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 构建命令行参数
    args = [
        'python', TRAIN_SCRIPT,
        f"--reins_mode={mode}",
        f"--reins_pos={param_dict['reins_pos']}",
        f"--reins_channels={param_dict['reins_channels']}",
        f"--save_dir={exp_dir}"
    ]
    print(f"\n[INFO] Running: {' '.join(args)}")
    # 启动训练
    subprocess.run(args, check=True)

    # 检查config.txt是否归档
    config_path = os.path.join(exp_dir, 'config.txt')
    if os.path.exists(config_path):
        print(f"[OK] config.txt archived: {config_path}")
    else:
        print(f"[WARN] config.txt not found in {exp_dir}")


def main():
    for exp in EXPERIMENTS:
        mode = exp['mode']
        for param in exp['params']:
            run_experiment(mode, param)

if __name__ == '__main__':
    main()
