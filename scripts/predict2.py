import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import os
import sys
import argparse
from typing import Tuple
import numpy as np
from tqdm import tqdm
from PIL import Image
from deeplab import DeeplabV3
from cityscapesscripts.helpers.labels import trainId2label
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import main as cityscapes_eval


def convert_trainId_to_labelId(pred_array: np.ndarray) -> np.ndarray:
    """将trainId转换为Cityscapes官方要求的labelId"""
    labelid_map = np.full_like(pred_array, 255, dtype=np.uint8)  # 默认填充255

    for train_id in np.unique(pred_array):
        if train_id == 255: continue
        if label_info := trainId2label.get( train_id ):
            labelid_map[pred_array == train_id] = label_info.id
    return labelid_map


def parse_cityscapes_name(voc_image_id: str) -> str:
    """将VOC格式ID转换为Cityscapes标准文件名"""
    # 输入示例: 2007_000032 -> 输出: aachen_000000_000019
    city_id, frame_id = voc_image_id.split('_')
    return f"{city_id.zfill(6)}_{frame_id.zfill(6)}_leftImg8bit.png"  # 符合官方命名规范


def validate_directory_structure():
    """验证预测目录结构是否符合官方要求"""
    required = [
        'leftImg8bit/val',
        'gtFine/val',
        'results/val'
    ]
    missing = [d for d in required if not os.path.exists(d)]
    if missing:
        raise FileNotFoundError(f"缺失关键目录: {missing}")

def parse_args() -> argparse.Namespace:
    # 使用argarse解析命令行参数
    parser = argparse.ArgumentParser(description="Cityscapes 评估脚本")
    parser.add_argument("--dataset-root", type=str, required=True,
                       help="数据集根目录（包含gtFine/leftImg8bit）")
    parser.add_argument("--pred-root", type=str, required=True,
                       help="预测结果输出目录")
    parser.add_argument( "--mode", type=int, choices=[0, 1, 2], default=0,
                         help="0=全流程, 1=仅预测, 2=仅评估" )
    parser.add_argument("--num-workers", type=int, default=4, help="并行工作线程数")
    return parser.parse_args()

def validate_paths(gt_dir: str, pred_dir: str) -> None:
    """增强路径校验逻辑"""
    required_files = [
        (os.path.join(gt_dir, "val"), "缺失验证集标签目录"),
        (os.path.join(pred_dir, "*.png"), "未找到预测结果文件")
    ]
    for path, msg in required_files:
        if not glob.glob(path):
            raise FileNotFoundError(f"{msg}: {path}")

def main():
    # 参数解析
    args = parse_args()

    # =====================================
    # 路径配置（替换原有硬编码路径）
    # =====================================
    gt_dir = os.path.join( args.dataset_root, "gtFine/val" )
    pred_dir = args.pred_root
    os.makedirs( pred_dir, exist_ok=True )
    validate_paths( gt_dir, pred_dir )

    # 预测模式
    if args.mode in [0, 1]:
        '''
        if not os.path.exists( pred_dir ):
            os.makedirs( pred_dir )
        '''
        deeplab = DeeplabV3()
        val_txt = os.path.join( args.dataset_root, "ImageSets/Segmentation/val.txt" )

        with open(val_txt) as f:
            for image_id in tqdm( f.read().splitlines(), desc="生成预测" ):
                image_path = os.path.join( args.dataset_root, "JPEGImages", f"{image_id}.jpg" )
                pred = deeplab.get_miou_png( Image.open( image_path ) )

                # 转换并保存结果
                labelId_map = convert_trainId_to_labelId( pred )
                save_name = parse_cityscapes_name( image_id )
                Image.fromarray( labelId_map ).save( os.path.join( pred_dir, save_name ) )

    # 评估模式
    if args.mode in [0, 2]:
        print( "\n启动Cityscapes官方评估..." )
        original_argv = sys.argv.copy()  # 备份原始参数
        sys.argv = [
            "",  # 占位argv[0]
            "--gt-root", gt_dir,
            "--pred-root", pred_dir,
            "--num-workers", str( args.num_workers )
        ]
        cityscapes_eval()

if __name__ == "__main__":
    main()