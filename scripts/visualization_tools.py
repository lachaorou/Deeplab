"""
visualization_tools.py

本脚本整合了原 visualize_loss_curves.py、visualize_samples.py、visualize_saved_checkpoints.py、plot_confusion_matrix.py、merge_loss_curves.py 等可视化相关功能。
每个功能以函数形式提供，便于统一调用和维护。
如需扩展更多可视化功能，可在本脚本中补充。

合并来源：
- visualize_loss_curves.py
- visualize_samples.py
- visualize_saved_checkpoints.py
- plot_confusion_matrix.py
- merge_loss_curves.py

用法示例：
    from visualization_tools import visualize_loss_curves, visualize_samples, visualize_saved_checkpoints, plot_confusion_matrix, merge_loss_curves
    visualize_loss_curves(...)
    visualize_samples(...)
    visualize_saved_checkpoints(...)
    plot_confusion_matrix(...)
    merge_loss_curves(...)
"""

# 以下为各原脚本功能的函数模板，具体实现可参考原脚本内容补充完善。

def visualize_loss_curves(loss_csv, val_loss_csv, output_path='loss_curve.png'):
    """
    可视化loss曲线，支持训练与验证loss对比。
    Args:
        loss_csv (str): 训练loss的csv文件路径
        val_loss_csv (str): 验证loss的csv文件路径
        output_path (str): 输出图片路径
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    loss_df = pd.read_csv(loss_csv)
    val_loss_df = pd.read_csv(val_loss_csv)
    plt.figure(figsize=(10, 6))
    plt.plot(loss_df['step'], loss_df['value'], label='Train Loss')
    plt.plot(val_loss_df['step'], val_loss_df['value'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Curve (断点续训自动拼接)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f'已生成 {output_path} 并弹出可视化窗口')

def visualize_samples(img_dir, gt_dir, pred_dir, out_dir, samples=5):
    """
    批量可视化样本输入、标签、预测结果。
    Args:
        img_dir (str): 原图目录
        gt_dir (str): 标签目录
        pred_dir (str): 预测结果目录
        out_dir (str): 输出目录
        samples (int): 可视化样本数
    """
    import os
    import random
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    from cityscapesscripts.helpers.labels import trainId2label
    os.makedirs(out_dir, exist_ok=True)
    palette = np.zeros((256, 3), dtype=np.uint8)
    for label in trainId2label:
        if label >= 0 and label < 19:
            palette[label] = trainId2label[label].color
    def colorize_mask(mask):
        return palette[mask]
    def vis_sample(img_name):
        img = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
        gt = Image.open(os.path.join(gt_dir, img_name.replace('.jpg', '.png')))
        base = img_name.replace('.jpg', '')
        if base.endswith('_gtFine_labelTrainIds'):
            base = base[:-len('_gtFine_labelTrainIds')]
        pred_name = base + '_gtFine_labelTrainIds.png'
        pred_path = os.path.join(pred_dir, pred_name)
        pred = Image.open(pred_path)
        gt_color = Image.fromarray(colorize_mask(np.array(gt)))
        pred_color = Image.fromarray(colorize_mask(np.array(pred)))
        vis = Image.new('RGB', (img.width * 3, img.height))
        vis.paste(img, (0, 0))
        vis.paste(gt_color, (img.width, 0))
        vis.paste(pred_color, (img.width * 2, 0))
        vis.save(os.path.join(out_dir, img_name))
    img_list = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    samples_list = random.sample(img_list, min(samples, len(img_list)))
    for img_name in samples_list:
        vis_sample(img_name)
    print(f'可视化已保存到 {out_dir}')

def visualize_saved_checkpoints(
    weights_dir,
    img_dir,
    gt_dir,
    val_txt,
    output_dir,
    num_classes=19,
    input_shape=(1024, 1536),
    backbone='mobilenet',
    num_samples=10,
    by_sample=False,
    cuda=False
):
    """
    可视化已保存权重的模型表现，支持按权重/按样本两种方式。
    Args:
        weights_dir (str): 权重文件目录
        img_dir (str): 验证集图片目录
        gt_dir (str): 验证集标签目录
        val_txt (str): 验证集ID列表txt
        output_dir (str): 输出可视化目录
        num_classes (int): 类别数
        input_shape (tuple): 输入分辨率
        backbone (str): 主干网络
        num_samples (int): 可视化样本数
        by_sample (bool): 是否按样本为主
        cuda (bool): 是否使用GPU
    """
    import os
    import random
    import torch
    from PIL import Image
    import numpy as np
    from tqdm import tqdm
    import cv2
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from models.deeplabv3_reins import DeepLab
    from utils.utils import cvtColor, preprocess_input, resize_image
    from utils.colorize import colorize_mask
    import torch.nn.functional as F

    def get_weight_files():
        files = [f for f in os.listdir(weights_dir) if f.endswith('.pth') and f.startswith('ep')]
        files = sorted(files, key=lambda x: int(x.split('-')[0].replace('ep','')))
        return files

    def load_model(weight_path):
        model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=8, pretrained=False)
        state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        if cuda:
            model.cuda()
        return model

    def get_miou_png(model, image, input_shape, cuda=False):
        image = cvtColor(image)
        orininal_h, orininal_w = np.array(image).shape[:2]
        image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if cuda:
                images = images.cuda()
            pr = model(images)[0]
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            pr = pr[int((input_shape[0] - nh) // 2) : int((input_shape[0] - nh) // 2 + nh),
                    int((input_shape[1] - nw) // 2) : int((input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)
        color_pred = colorize_mask(pr)
        return Image.fromarray(color_pred)

    # 读取验证集样本ID
    with open(val_txt, 'r') as f:
        val_ids = [x.strip() for x in f.readlines() if x.strip()]
    random.seed(42)
    sample_ids = random.sample(val_ids, min(num_samples, len(val_ids)))
    weight_files = get_weight_files()

    if not by_sample:
        # 按权重可视化
        for weight_file in weight_files:
            epoch_str = weight_file.split('-')[0]
            out_dir = os.path.join(output_dir, f'val_compare_{epoch_str}')
            os.makedirs(out_dir, exist_ok=True)
            model = load_model(os.path.join(weights_dir, weight_file))
            rows = []
            for img_id in tqdm(sample_ids, desc=f'Visualizing {epoch_str}'):
                img_path = os.path.join(img_dir, img_id + '.jpg')
                gt_path = os.path.join(gt_dir, img_id + '.png')
                pred_path = os.path.join(out_dir, img_id + '_pred.png')
                img = Image.open(img_path)
                gt = Image.open(gt_path)
                pred = get_miou_png(model, img, input_shape, cuda)
                pred.save(pred_path)
                img_save_path = os.path.join(out_dir, img_id + '_img.jpg')
                img.save(img_save_path)
                gt_gray_save_path = os.path.join(out_dir, img_id + '_gt_gray.png')
                gt.save(gt_gray_save_path)
                gt_color = colorize_mask(np.array(gt))
                gt_color_img = Image.fromarray(gt_color)
                gt_color_save_path = os.path.join(out_dir, img_id + '_gt_color.png')
                gt_color_img.save(gt_color_save_path)
                abs_img = os.path.abspath(img_save_path).replace('\\', '/')
                abs_gt_color = os.path.abspath(gt_color_save_path).replace('\\', '/')
                abs_gt_gray = os.path.abspath(gt_gray_save_path).replace('\\', '/')
                abs_pred = os.path.abspath(pred_path).replace('\\', '/')
                rows.append(f'| {img_id} | ![]({abs_img}) | ![]({abs_gt_color}) | ![]({abs_gt_gray}) | ![]({abs_pred}) |')
            md_path = os.path.join(out_dir, 'compare.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write('| ID | 原图 | 标签(彩色) | 标签(灰度) | 预测 |\n')
                f.write('|----|------|-----------|-----------|------|\n')
                for row in rows:
                    f.write(row + '\n')
            print(f'可视化结果已保存到: {out_dir}')
    else:
        # 按样本可视化
        out_root = os.path.join(output_dir, 'by_sample')
        os.makedirs(out_root, exist_ok=True)
        models = {}
        for weight_file in weight_files:
            epoch_str = weight_file.split('-')[0]
            model = load_model(os.path.join(weights_dir, weight_file))
            models[epoch_str] = model
        for img_id in tqdm(sample_ids, desc='By-sample visualization'):
            sample_dir = os.path.join(out_root, img_id)
            os.makedirs(sample_dir, exist_ok=True)
            img_path = os.path.join(img_dir, img_id + '.jpg')
            gt_path = os.path.join(gt_dir, img_id + '.png')
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            img_save_path = os.path.join(sample_dir, img_id + '_img.jpg')
            img.save(img_save_path)
            gt_gray_save_path = os.path.join(sample_dir, img_id + '_gt_gray.png')
            gt.save(gt_gray_save_path)
            gt_color = colorize_mask(np.array(gt))
            gt_color_img = Image.fromarray(gt_color)
            gt_color_save_path = os.path.join(sample_dir, img_id + '_gt_color.png')
            gt_color_img.save(gt_color_save_path)
            pred_paths = []
            for epoch_str, model in models.items():
                pred = get_miou_png(model, img, input_shape, cuda)
                pred_name = f'{epoch_str}_pred.png'
                pred_path = os.path.join(sample_dir, pred_name)
                pred.save(pred_path)
                pred_paths.append((epoch_str, pred_path))
            abs_img = os.path.abspath(img_save_path).replace('\\', '/')
            abs_gt_color = os.path.abspath(gt_color_save_path).replace('\\', '/')
            abs_gt_gray = os.path.abspath(gt_gray_save_path).replace('\\', '/')
            md_path = os.path.join(sample_dir, 'compare.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f'# 样本: {img_id}\n')
                f.write('| 原图 | 标签(彩色) | 标签(灰度) |')
                for epoch_str, _ in pred_paths:
                    f.write(f' {epoch_str}_预测 |')
                f.write('\n|------|-----------|-----------|' + '------|'*len(pred_paths) + '\n')
                f.write(f'| ![]({abs_img}) | ![]({abs_gt_color}) | ![]({abs_gt_gray}) |')
                for _, pred_path in pred_paths:
                    abs_pred = os.path.abspath(pred_path).replace('\\', '/')
                    f.write(f' ![]({abs_pred}) |')
                f.write('\n')
        print(f'按样本可视化结果已保存到: {out_root}')

def plot_confusion_matrix(csv_path, out_path):
    """
    绘制混淆矩阵热力图。
    Args:
        csv_path (str): 混淆矩阵csv文件路径
        out_path (str): 输出图片路径
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = pd.read_csv(csv_path, index_col=0)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f'混淆矩阵热力图已保存到: {out_path}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeeplabV3+ 可视化工具集")
    subparsers = parser.add_subparsers(dest='command')

    # loss曲线
    parser_loss = subparsers.add_parser('loss', help='可视化loss曲线')
    parser_loss.add_argument('--loss_csv', type=str, required=True)
    parser_loss.add_argument('--val_loss_csv', type=str, required=True)
    parser_loss.add_argument('--output', type=str, default='loss_curve.png')

    # 样本可视化
    parser_samples = subparsers.add_parser('samples', help='批量可视化样本')
    parser_samples.add_argument('--img_dir', type=str, required=True)
    parser_samples.add_argument('--gt_dir', type=str, required=True)
    parser_samples.add_argument('--pred_dir', type=str, required=True)
    parser_samples.add_argument('--out_dir', type=str, required=True)
    parser_samples.add_argument('--samples', type=int, default=5)

    # 权重可视化
    parser_ckpt = subparsers.add_parser('checkpoints', help='可视化已保存权重表现')
    parser_ckpt.add_argument('--weights_dir', type=str, required=True)
    parser_ckpt.add_argument('--img_dir', type=str, required=True)
    parser_ckpt.add_argument('--gt_dir', type=str, required=True)
    parser_ckpt.add_argument('--val_txt', type=str, required=True)
    parser_ckpt.add_argument('--output_dir', type=str, required=True)
    parser_ckpt.add_argument('--num_classes', type=int, default=19)
    parser_ckpt.add_argument('--input_shape', type=int, nargs=2, default=[1024,1536])
    parser_ckpt.add_argument('--backbone', type=str, default='mobilenet')
    parser_ckpt.add_argument('--num_samples', type=int, default=10)
    parser_ckpt.add_argument('--by_sample', action='store_true')
    parser_ckpt.add_argument('--cuda', action='store_true')

    # 混淆矩阵
    parser_cm = subparsers.add_parser('confusion', help='绘制混淆矩阵热力图')
    parser_cm.add_argument('--csv', type=str, required=True)
    parser_cm.add_argument('--out', type=str, required=True)

    args = parser.parse_args()
    if args.command == 'loss':
        visualize_loss_curves(args.loss_csv, args.val_loss_csv, args.output)
    elif args.command == 'samples':
        visualize_samples(args.img_dir, args.gt_dir, args.pred_dir, args.out_dir, args.samples)
    elif args.command == 'checkpoints':
        visualize_saved_checkpoints(
            weights_dir=args.weights_dir,
            img_dir=args.img_dir,
            gt_dir=args.gt_dir,
            val_txt=args.val_txt,
            output_dir=args.output_dir,
            num_classes=args.num_classes,
            input_shape=tuple(args.input_shape),
            backbone=args.backbone,
            num_samples=args.num_samples,
            by_sample=args.by_sample,
            cuda=args.cuda
        )
    elif args.command == 'confusion':
        plot_confusion_matrix(args.csv, args.out)
    else:
        parser.print_help()
