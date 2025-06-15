"""
批量运行MobileNetV2+Reins多点插入消融实验脚本
- 支持多点插入（如backbone、aspp、multi）
- 自动归档参数与结果
"""
import os
import subprocess
from datetime import datetime

EXPERIMENTS = [
    # 多点插入组合
    {'reins_mode': 'multi', 'multi_reins_points': ['backbone', 'aspp'], 'token_length': 64, 'embed_dims': 320, 'num_layers': 2},
    {'reins_mode': 'multi', 'multi_reins_points': ['backbone', 'aspp', 'shortcut'], 'token_length': 64, 'embed_dims': 320, 'num_layers': 3},
    # 单点插入对比
    {'reins_mode': 'single', 'reins_pos': 'aspp', 'token_length': 64, 'embed_dims': 320, 'num_layers': 1},
    {'reins_mode': 'single', 'reins_pos': 'backbone', 'token_length': 64, 'embed_dims': 320, 'num_layers': 1},
    # 无Reins基线
    {'reins_mode': 'none'},
]

TRAIN_SCRIPT = os.path.join('scripts', 'train.py')
RESULTS_DIR = os.path.join('results', 'multi_reins_experiments')


def run_experiment(params):
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"reins_{params.get('reins_mode')}_{'_'.join(params.get('multi_reins_points', [params.get('reins_pos','none')]))}_{time_str}"
    exp_dir = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    args = [
        'python', TRAIN_SCRIPT,
        '--backbone', 'mobilenet',
        f"--save_dir={exp_dir}"
    ]
    if params['reins_mode'] == 'multi':
        args += [f"--reins_mode=multi", f"--multi_reins_points={'/'.join(params['multi_reins_points'])}", f"--token_length={params['token_length']}", f"--embed_dims={params['embed_dims']}", f"--num_layers={params['num_layers']}"]
    elif params['reins_mode'] == 'single':
        args += [f"--reins_mode=single", f"--reins_pos={params['reins_pos']}", f"--token_length={params['token_length']}", f"--embed_dims={params['embed_dims']}", f"--num_layers={params['num_layers']}"]
    # 无Reins基线
    print(f"\n[INFO] Running: {' '.join(args)}")
    subprocess.run(args, check=True)
    config_path = os.path.join(exp_dir, 'config.txt')
    if os.path.exists(config_path):
        print(f"[OK] config.txt archived: {config_path}")
    else:
        print(f"[WARN] config.txt not found in {exp_dir}")

def main():
    for params in EXPERIMENTS:
        run_experiment(params)

if __name__ == '__main__':
    main()
