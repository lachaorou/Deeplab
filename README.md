# Deeplab

---

## 目录结构与数据集扩充建议（结合官方与VOCdevkit）

### 推荐目录结构

VOCdevkit/
├─ JPEGImages/                # 训练图片（建议2975张，官方Cityscapes同量）
├─ SegmentationClass/         # 标签图片（与JPEGImages一一对应）
├─ ImageSets/
│   └─ Segmentation/
│        ├─ train.txt         # 训练集图片名列表（2975行）
│        └─ val.txt           # 验证集图片名列表（500行）

```
VOCdevkit/
├─ JPEGImages/                # 训练图片（建议2975张，官方Cityscapes同量）
├─ SegmentationClass/         # 标签图片（与JPEGImages一一对应）
├─ ImageSets/
│   └─ Segmentation/
│        ├─ train.txt         # 训练集图片名列表（2975行）
│        └─ val.txt           # 验证集图片名列表（500行）
```

### 训练与评估输出目录

```
results/
├─ logs_deeplab/           # 训练权重、loss曲线、日志自动归档目录
│    └─ archive_20250607_153000/logs_mobilenetv2rein/...
├─ mious_deeplab/          # mIoU评估输出、可视化、detection-results自动归档目录
│    └─ archive_20250607_153000/miou_out_mobilenetv2_rein/...
```

> 推荐所有训练权重、loss曲线、日志、mIoU评估结果均通过 auto/auto_archive_logs_and_miou.py 自动归档到 results/logs_deeplab/ 和 results/mious_deeplab/，便于管理和对比。

### 训练参数建议
- 推荐训练总轮数：100轮（epoch=100），每5~10轮保存一次权重和评估一次mIoU。
- batch_size、input_shape等参数根据显存适配。
- 训练和验证集数量建议与官方Cityscapes一致（2975/500），提升泛化能力。

### 数据集扩充操作
- 如当前VOCdevkit/JPEGImages/下图片少于2975张，建议补全至官方数量。
- 标签SegmentationClass/需与JPEGImages/一一对应，像素值分布需用check脚本排查。
- train.txt、val.txt需与图片实际数量对应。

---

## 训练步骤 How2train

1. 将VOC格式数据集放入 data/voc/VOCdevkit/ 下。
2. 推荐运行 auto/auto_cityscapes2voc.py 进行格式转换和主名txt生成。
3. 在 train.py 中设置参数，推荐如下：
   ```python
   --model_path results/logs_deeplab/logs_mobilenetv2rein/best_epoch_weights.pth
   --log_dir results/logs_deeplab/logs_mobilenetv2rein/
   --backbone mobilenet
   --num_classes 19
   --epoch 100
   # 其它参数...
   ```
4. 运行 train.py 进行训练。
5. 训练完成后，权重和日志会自动保存在 results/logs_deeplab/ 下。
6. 可定期运行 auto/auto_archive_logs_and_miou.py 进行归档。

## 评估步骤 miou

1. 在 get_miou.py 中设置参数，推荐如下：
   ```python
   --model_path results/logs_deeplab/logs_mobilenetv2rein/best_epoch_weights.pth
   --miou_out_path results/mious_deeplab/miou_out_mobilenetv2_rein/
   --num_classes 19
   # 其它参数...
   ```
2. 运行 get_miou.py 进行评估，mIoU结果、混淆矩阵、可视化等会保存在 results/mious_deeplab/ 下。
3. 可定期运行 auto/auto_archive_logs_and_miou.py 进行归档。

> 其余训练、预测、评估脚本参数请参考 scripts/ 目录下各脚本说明。

## Top News
**`2022-04`**:**支持多GPU训练。**

**`2022-03`**:**进行大幅度更新、支持step、cos学习率下降法、支持adam、sgd优化器选择、支持学习率根据batch_size自适应调整。**
BiliBili视频中的原仓库地址为：https://github.com/bubbliiiing/deeplabv3-plus-pytorch/tree/bilibili

**`2020-08`**:**创建仓库、支持多backbone、支持数据miou评估、标注数据处理、大量注释等。**

## 相关仓库
| 模型       | 路径                                                  |
| :--------- | :---------------------------------------------------- |
| Unet       | https://github.com/bubbliiiing/unet-pytorch           |
| PSPnet     | https://github.com/bubbliiiing/pspnet-pytorch         |
| deeplabv3+ | https://github.com/bubbliiiing/deeplabv3-plus-pytorch |
| hrnet      | https://github.com/bubbliiiing/hrnet-pytorch          |

### 性能情况
| 训练数据集 |                                                          权值文件名称                                                           | 测试数据集 | 输入图片大小 | mIOU  |
| :--------: | :-----------------------------------------------------------------------------------------------------------------------------: | :--------: | :----------: | :---: |
| VOC12+SBD  | [deeplab_mobilenetv2.pth](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/deeplab_mobilenetv2.pth) | VOC-Val12  |   512x512    | 72.59 |
| VOC12+SBD  |    [deeplab_xception.pth](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/deeplab_xception.pth)    | VOC-Val12  |   512x512    | 76.95 |

### 所需环境
torch==1.2.0

### 注意事项
代码中的deeplab_mobilenetv2.pth和deeplab_xception.pth是基于VOC拓展数据集训练的。训练和预测时注意修改backbone。

### 文件下载
训练所需的deeplab_mobilenetv2.pth和deeplab_xception.pth可在百度网盘中下载。
链接: https://pan.baidu.com/s/1IQ3XYW-yRWQAy7jxCUHq8Q 提取码: qqq4

VOC拓展数据集的百度网盘如下：
链接: https://pan.baidu.com/s/1vkk3lMheUm6IjTXznlg7Ng 提取码: 44mk

---

## 目录结构

Deeplabv3plus/
├─ Automation/
│    ├─ AutoArchiveLogsAndMiou.py
│    ├─ AutoArchiveResults.py
│    ├─ AutoCityscapes2Voc.py
│    ├─ AutoReorganizeProject.py
│    ├─ AutoSessionLog.py
│    └─ Readme.txt
├─ Inspect/
│    ├─ CheckAndFixLabelValues.py
│    ├─ CheckLabelPredPair.py
│    ├─ CheckLabelValues.py
│    ├─ CheckMiouData.py
│    ├─ CheckPredUnique.py
│    ├─ VisualizeLabelPred.py
│    ├─ VisualizeCompare.py
│    └─ Readme.txt
├─ Utils/
│    ├─ Callbacks.py
│    ├─ CheckLabelValues.py
│    ├─ CheckMissingLabels.py
│    ├─ Colorize.py
│    ├─ Dataloader0.py
│    ├─ StatLabelDistribution.py
│    ├─ TrainUtils.py
│    ├─ UtilsFit.py
│    ├─ UtilsMetrics.py
│    └─ Utils.py
├─ Dataset/
│    ├─ Cityscapes/
│    ├─ ModelData/
│    ├─ Processed/
│    ├─ Scripts/
│    ├─ Voc/
│    └─ Readme.txt
├─ Models/
│    ├─ Deeplab.py
│    ├─ Deeplabv3Plus.py
│    ├─ Deeplabv3Reins.py
│    ├─ Deeplabv3Training.py
│    ├─ Mobilenetv2.py
│    ├─ Reins.py
│    ├─ Xception.py
│    └─ ModelData/
├─ Scripts/
│    ├─ AutoResumeTrain.py
│    ├─ AutoSplitTrainVal.py
│    ├─ CheckLabelValidity.py
│    ├─ CleanValTxt.py
│    ├─ GetMiou.py
│    ├─ ModelSummary.py
│    ├─ Predict.py
│    ├─ Predict2.py
│    ├─ SelectValCoverAllClasses.py
│    ├─ Train.py
│    └─ Readme.txt
├─ Results/
│    ├─ Logs说明.md
│    ├─ Miou说明.md
│    ├─ LogsDeeplab/
│    └─ MiousDeeplab/
├─ Documents/
│    ├─ DeeplabNotes/
│    ├─ BackupMap.txt
│    ├─ CopilotWelcome.txt
│    ├─ Readme.txt
│    └─ ...
├─ Cityscapescripts/
│    └─ ...（官方评测包）
├─ License
├─ Readme.md
├─ Requirements.txt
└─ Config.py
