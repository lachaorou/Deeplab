import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
from tqdm import tqdm
from models.deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import main as eval_main
import numpy as np
import argparse
import glob
import pandas as pd
import datetime


'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeeplabV3+ mIoU Evaluation")
    parser.add_argument('--experiment_csv', type=str, default='Documents/experiment_records.csv', help='实验参数表格csv路径')
    args = parser.parse_args()

    # 只从参数表读取参数
    df = pd.read_csv(args.experiment_csv)
    print("可选实验版本：")
    for idx, row in df.iterrows():
        print(f"{idx}: {row['model_name']} | {row['train_time']} | {row['backbone']} | {row['input_shape']} | {row['weight_path']}")
    choice = int(input("请输入要评估的实验序号："))
    params = df.iloc[choice]
    # 自动引用参数
    VOCdevkit_path = params['voc_path']
    input_shape = eval(params['input_shape'])
    backbone = params['backbone']
    model_path = params['weight_path']
    num_classes = int(params['num_classes']) if 'num_classes' in params else 19
    model_name = params['model_name']
    train_time = params['train_time']
    # 自动生成评估日期
    eval_date = datetime.datetime.now().strftime('%Y_%m_%d')
    # 自动生成唯一评估输出目录
    miou_out_path = f"results/mious_deeplab/miou_out_{model_name}_{train_time}_eval_{eval_date}"
    print(f"[Auto] 评估输出目录: {miou_out_path}")
    if not os.path.exists(miou_out_path):
        os.makedirs(miou_out_path)

    # 重新赋值 image_ids，确保路径正确
    val_txt_path = os.path.join(VOCdevkit_path, "ImageSets", "Segmentation", "val.txt")
    print("val.txt 路径：", val_txt_path)
    image_ids = open(val_txt_path, 'r').read().splitlines()

    pred_dir = os.path.join(miou_out_path, 'detection-results')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    # 自动推断标签目录
    gt_dir = os.path.join(VOCdevkit_path, 'SegmentationClass')

    # 自动同步 token_length，embed_dims，其它参数
    # 优先从权重文件自动读取 token_length，确保与权重一致
    import torch
    def extract_token_length_from_state_dict(sd):
        # 兼容 DataParallel 及嵌套结构
        if 'reins.learnable_tokens' in sd:
            return sd['reins.learnable_tokens'].shape[1]
        elif 'module.reins.learnable_tokens' in sd:
            return sd['module.reins.learnable_tokens'].shape[1]
        elif 'state_dict' in sd:
            return extract_token_length_from_state_dict(sd['state_dict'])
        else:
            return None
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        token_length_from_weight = extract_token_length_from_state_dict(state_dict)
    except Exception as e:
        print(f"[Warning] 权重文件读取失败，token_length 自动同步跳过: {e}")
        token_length_from_weight = None

    # 强制参数表为唯一源，必须有 token_length
    if 'token_length' not in params or pd.isnull(params['token_length']):
        raise ValueError('参数表缺失 token_length，请补全 experiment_records.csv！')
    if 'num_layers' not in params or pd.isnull(params['num_layers']):
        raise ValueError('参数表缺失 num_layers，请补全 experiment_records.csv！')
    token_length = int(params['token_length'])
    num_layers = int(params['num_layers'])
    img_dir = params['img_dir'] if 'img_dir' in params else 'Dataset/Voc/VOCdevkit/JPEGImages'

    print("Load model.")
    deeplab = DeeplabV3(
        num_classes=num_classes,
        model_path=model_path,
        backbone=backbone,
        input_shape=input_shape,
        use_tokens=True,
        token_length=token_length,
        num_layers=num_layers
    )
    print("Load model done.")
    print("Configurations:")
    print("----------------------------------------------------------------------")
    print(f"|{'model_path':>22} | {str(model_path):>40}|")
    print(f"|{'miou_out_path':>22} | {str(miou_out_path):>40}|")
    print(f"|{'num_classes':>22} | {str(num_classes):>40}|")
    print(f"|{'backbone':>22} | {str(backbone):>40}|")
    print(f"|{'input_shape':>22} | {str(input_shape):>40}|")
    print(f"|{'VOCdevkit_path':>22} | {str(VOCdevkit_path):>40}|")
    print("----------------------------------------------------------------------")

    # Cityscapes官方19类
    name_classes = [
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
        "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
    ]

    print("Get predict result.")
    # 可视化前10张图片的输入、标签、预测
    import matplotlib.pyplot as plt
    vis_num = 10
    for idx, image_id in enumerate(tqdm(image_ids)):
        image_path  = os.path.join(img_dir, image_id+".jpg")
        label_path  = os.path.join(gt_dir, image_id+".png")
        image       = Image.open(image_path)
        label       = np.array(Image.open(label_path))
        pred_img    = deeplab.get_miou_png(image)
        pred_np     = np.array(pred_img)
        if idx < 2:
            pass
        # 保存预测结果
        pred_img.save(os.path.join(pred_dir, image_id + ".png"))
        # 可视化前10张
        if idx < vis_num:
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.imshow(image)
            plt.title('Input')
            plt.axis('off')
            plt.subplot(1,3,2)
            plt.imshow(label, cmap='tab20')
            plt.title('Label')
            plt.axis('off')
            plt.subplot(1,3,3)
            plt.imshow(pred_np, cmap='tab20')
            plt.title('Pred')
            plt.axis('off')
            vis_path = os.path.join(miou_out_path, f'vis_{image_id}.png')
            plt.savefig(vis_path)
            plt.close()
    print("Get predict result done.")

    debug = False  # 可根据需要设为 True 以输出详细调试信息
    print("Calculate miou.")
    # 自动适配 compute_mIoU 返回值数量
    miou_result = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes, debug=debug)
    hist, IoUs, PA_Recall, Precision = miou_result[:4]
    # 可选：如需处理异常统计，可用 miou_result[4:]，如 abnormal_pairs, abnormal_types = miou_result[4:6]
    show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)