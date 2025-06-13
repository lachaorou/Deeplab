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


'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeeplabV3+ mIoU Evaluation")
    parser.add_argument('--img_dir', type=str, default='data/voc/VOCdevkit/JPEGImages', help='图片目录，需与val.txt图片名对应')
    parser.add_argument('--debug', action='store_true', help='是否开启详细debug信息')
    parser.add_argument('--model_path', type=str, default='results/logs_deeplab/logs_mobilenetv2rein/best_epoch_weights.pth', help='模型权重路径')
    parser.add_argument('--miou_out_path', type=str, default='results/mious_deeplab/miou_out_mobilenetv2_rein', help='mIoU评估输出目录')
    args = parser.parse_args()
    img_dir = args.img_dir
    debug = args.debug
    model_path = args.model_path
    miou_out_path = args.miou_out_path
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   分类个数，Cityscapes官方19类
    #------------------------------#
    num_classes     = 19
    #--------------------------------------------#
    #   区分的种类，严格与trainId顺序一致
    #--------------------------------------------#
    name_classes    = [
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
        "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
    ]
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = 'data/voc/VOCdevkit'

    input_shape = [896, 640]  # 与训练保持一致
    backbone = 'mobilenet'

    image_ids       = open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/val.txt"),'r').read().splitlines()
    gt_dir          = os.path.join(VOCdevkit_path, "SegmentationClass/")
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        print("Load model.")
        deeplab = DeeplabV3(
            num_classes=num_classes,  # 保持和训练一致
            model_path=model_path,    # 权重路径
            backbone=backbone,
            input_shape=input_shape,
            use_tokens=True
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

    # 只在单独评估脚本中开启详细debug
    if miou_mode == 0 or miou_mode == 2:
        print("Calculate miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes, debug=debug)
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)