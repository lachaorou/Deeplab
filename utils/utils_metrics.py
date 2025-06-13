# print("[Debug] utils_metrics.py loaded from:", __file__)
import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def f_score(inputs, target, beta=1, smooth = 1e-5, threshold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threshold).float()
    tp = torch.sum(temp_target * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs, axis=[0,1]) - tp
    fn = torch.sum(temp_target, axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None, debug=False):
    if debug:
        print('[Debug] compute_mIoU called')
        print('[Debug] gt_dir:', gt_dir)
        print('[Debug] pred_dir:', pred_dir)
        print('[Debug] png_name_list:', png_name_list)
        print('[Debug] num_classes:', num_classes)
    hist = np.zeros((num_classes, num_classes))
    gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]
    if len(gt_imgs) != len(pred_imgs):
        print(f"[Check][Error] 标签图片数 {len(gt_imgs)} 与预测图片数 {len(pred_imgs)} 不一致！")
    abnormal_pairs = []
    abnormal_types = {}  # 新增：异常类型统计
    for ind in range(len(gt_imgs)):
        try:
            pred = np.array(Image.open(pred_imgs[ind]))  
            label = np.array(Image.open(gt_imgs[ind]))  
        except Exception as e:
            if debug:
                print(f"[Check][Error] 第{ind}对图片读取失败: {gt_imgs[ind]}, {pred_imgs[ind]}, 错误: {e}")
            abnormal_pairs.append((gt_imgs[ind], pred_imgs[ind], 'read_error'))
            abnormal_types.setdefault('read_error', []).append((gt_imgs[ind], pred_imgs[ind]))
            continue
        if label.shape != pred.shape:
            if debug:
                print(f"[Check][Error] shape不一致: {gt_imgs[ind]} {label.shape} vs {pred_imgs[ind]} {pred.shape}")
            abnormal_pairs.append((gt_imgs[ind], pred_imgs[ind], 'shape'))
            abnormal_types.setdefault('shape', []).append((gt_imgs[ind], pred_imgs[ind]))
            continue
        label_unique = np.unique(label)
        pred_unique = np.unique(pred)
        if np.any(label_unique < 0) or np.any(label_unique >= num_classes):
            abnormal_pairs.append((gt_imgs[ind], pred_imgs[ind], 'label_unique'))
            abnormal_types.setdefault('label_unique', []).append((gt_imgs[ind], pred_imgs[ind]))
        if np.any(pred_unique < 0) or np.any(pred_unique >= num_classes):
            abnormal_pairs.append((gt_imgs[ind], pred_imgs[ind], 'pred_unique'))
            abnormal_types.setdefault('pred_unique', []).append((gt_imgs[ind], pred_imgs[ind]))
        if debug:
            print(f'[Debug][{ind}] label unique:', label_unique, 'pred unique:', pred_unique)
        if len(label.flatten()) != len(pred.flatten()):  
            if debug:
                print(
                    'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                        len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                        pred_imgs[ind]))
            abnormal_pairs.append((gt_imgs[ind], pred_imgs[ind], 'length_mismatch'))
            abnormal_types.setdefault('length_mismatch', []).append((gt_imgs[ind], pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)  
    if abnormal_pairs:
        print(f"[Check][Summary] 检查到异常图片对: {abnormal_pairs}")
    IoUs        = per_class_iu(hist)
    PA_Recall   = per_class_PA_Recall(hist)
    Precision   = per_class_Precision(hist)
    if debug:
        print('[Debug] hist matrix:', hist)
    # 新增：返回异常类型统计
    return np.array(hist, int), IoUs, PA_Recall, Precision, abnormal_pairs, abnormal_types

def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size = 12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = True)
    # print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    # print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
    
    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    # print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    # print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))

# Cityscapes官方19类配色（顺序严格对应类别0~18）
CITYSCAPES_COLORMAP = [
    (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153),
    (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
    ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
    (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32)
]

def colorize_mask(mask, colormap=CITYSCAPES_COLORMAP):
    """
    将0~18的mask可视化为Cityscapes官方配色，255为ignore设为黑色。
    mask: HxW, np.uint8
    return: HxWx3, np.uint8
    """
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(colormap):
        color_mask[mask == label] = color
    color_mask[mask == 255] = (0, 0, 0)  # ignore设为黑色
    return color_mask
