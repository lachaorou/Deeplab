# Deeplabv3+ 主干网络对比实验全流程手册（含论文/代码/数据集/社区）

## 1. 经典主干网络与论文

| 主干网络      | 论文/年份 | 论文链接 | 适配 Deeplabv3+ | 轻量化 | 精度 | 备注 |
|---------------|-----------|----------|-----------------|--------|------|------|
| Xception      | 2017      | [paper](https://arxiv.org/abs/1610.02357) | 官方推荐 | ✗ | ★★★★★ | 深度可分离卷积 |
| MobileNetV2   | 2018      | [paper](https://arxiv.org/abs/1801.04381) | 官方推荐 | ★★★★★ | ★★☆☆☆ | 轻量化 |
| MobileNetV3   | 2019      | [paper](https://arxiv.org/abs/1905.02244) | 社区支持 | ★★★★★ | ★★☆☆☆ | 轻量化 |
| EfficientNet  | 2019      | [paper](https://arxiv.org/abs/1905.11946) | 社区支持 | ★★★★☆ | ★★★★☆ | 轻量高效 |
| ConvNeXt      | 2022      | [paper](https://arxiv.org/abs/2201.03545) | 社区支持 | ★★★☆☆ | ★★★★★ | SOTA |
| Swin Transformer | 2021   | [paper](https://arxiv.org/abs/2103.14030) | 社区支持 | ★★☆☆☆ | ★★★★★ | Transformer |
| SegFormer     | 2021      | [paper](https://arxiv.org/abs/2105.15203) | 社区支持 | ★★★★☆ | ★★★★★ | 轻量+高精度 |
| InternImage   | 2023      | [paper](https://arxiv.org/abs/2303.05631) | 社区支持 | ★★☆☆☆ | ★★★★★ | SOTA |

---

## 2. 经典 GitHub 实现与权重

| 主干 | 官方/经典实现 | 预训练权重 | 适配 Deeplabv3+ |
|------|--------------|------------|-----------------|
| Xception | [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/deeplab) <br> [pytorch-segmentation](https://github.com/yassouali/pytorch-segmentation) | [GoogleDrive](https://drive.google.com/drive/folders/1G6vFq6Qh2QkQwQn6QwQn6QwQn6QwQn6Q) | 已集成 |
| MobileNetV2 | [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/deeplab) <br> [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception) | [GoogleDrive](https://drive.google.com/drive/folders/1G6vFq6Qh2QkQwQn6QwQn6QwQn6QwQn6Q) | 已集成 |
| MobileNetV3 | [timm](https://github.com/huggingface/pytorch-image-models) | [timm权重](https://github.com/huggingface/pytorch-image-models#pretrained-weights) | timm支持 |
| EfficientNet | [timm](https://github.com/huggingface/pytorch-image-models) <br> [rwightman/efficientnet-pytorch](https://github.com/rwightman/efficientnet-pytorch) | [timm权重](https://github.com/huggingface/pytorch-image-models#pretrained-weights) | timm支持 |
| ConvNeXt | [timm](https://github.com/huggingface/pytorch-image-models) | [timm权重](https://github.com/huggingface/pytorch-image-models#pretrained-weights) | timm支持 |
| Swin Transformer | [Swin-Transformer-Semantic-Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation) | [权重](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation#model-zoo) | 需适配 |
| SegFormer | [NVIDIA/SegFormer](https://github.com/NVlabs/SegFormer) | [权重](https://github.com/NVlabs/SegFormer#pre-trained-models) | 需适配 |
| InternImage | [OpenMMLab/InternImage](https://github.com/OpenGVLab/InternImage) | [权重](https://github.com/OpenGVLab/InternImage#model-zoo) | 需适配 |

---

## 3. Cityscapes 数据集与社区讨论

- 官网：[Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- 论文：[The Cityscapes Dataset for Semantic Urban Scene Understanding](https://www.cityscapes-dataset.com/wordpress/wp-content/papers/2016_CVPR_cityscapes.pdf)
- 公开 benchmark 排行榜：[Cityscapes Leaderboard](https://www.cityscapes-dataset.com/benchmarks/)
- 经典分割方法对比与讨论（知乎/论坛/issue）：
  - [知乎：Cityscapes 语义分割方法对比](https://zhuanlan.zhihu.com/p/349073282)
  - [GitHub Issue 讨论](https://github.com/NVlabs/SegFormer/issues?q=cityscapes)
  - [Papers With Code - Cityscapes Semantic Segmentation](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes)

---

## 4. 典型实验流程（详细到每一步）

1. **环境准备**
   - 安装 PyTorch、timm、mmcv、opencv、cityscapesscripts 等依赖
   - 下载 Cityscapes 数据集并解压到 Dataset/ 目录

2. **主干网络集成**
   - 轻量化主干：MobileNetV2（已集成）、pip install timm 后可用 MobileNetV3、EfficientNet、MobileViT
   - 高精度主干：Xception（已集成）、pip install timm 后可用 ConvNeXt、Swin Transformer、SegFormer
   - 参考 timm 官方文档：[timm models](https://huggingface.co/docs/timm/index)

3. **权重准备**
   - 下载对应主干的 ImageNet 预训练权重，放入 models/model_data/
   - 权重命名与 train.py 加载逻辑保持一致

4. **训练脚本参数设置**
   - backbone 参数支持多主干选择
   - batch_size、lr、optimizer 按主干推荐设置
   - 训练命令示例：
     ```sh
     python scripts/train.py --backbone mobilenetv2 --epochs 100 --batch_size 16
     python scripts/train.py --backbone xception --epochs 100 --batch_size 8
     python scripts/train.py --backbone convnext_tiny --epochs 100 --batch_size 8
     ```

5. **评估与可视化**
   - 运行 get_miou.py 评估各模型
   - 记录 mIoU、推理速度、参数量
   - 可视化结果自动归档到 Results/，并生成对比表格

6. **结果归档与对比**
   - 归档所有实验结果到 Results/ 下，命名如 logs_xxx_20250613/
   - 用 pandas 生成 csv/markdown 对比表格

7. **社区最佳实践与经验**
   - 参考 Papers With Code、知乎、GitHub issue 讨论最新主干和分割方法
   - 关注 timm、mmsegmentation 等库的更新

---

## 5. 推荐主干网络对比表（2024-2025）

| 主干         | 轻量化 | 精度 | timm支持 | 适配难度 | 论文/代码 | Cityscapes SOTA | 备注 |
|--------------|--------|------|----------|----------|-----------|-----------------|------|
| MobileNetV2  | ★★★★★ | ★★☆☆☆ | 是 | 低 | [paper](https://arxiv.org/abs/1801.04381) [code](https://github.com/jfzhang95/pytorch-deeplab-xception) | 否 | 已集成 |
| MobileNetV3  | ★★★★★ | ★★☆☆☆ | 是 | 低 | [paper](https://arxiv.org/abs/1905.02244) [timm](https://github.com/huggingface/pytorch-image-models) | 否 | timm |
| EfficientNet | ★★★★☆ | ★★★★☆ | 是 | 低 | [paper](https://arxiv.org/abs/1905.11946) [timm](https://github.com/huggingface/pytorch-image-models) | 否 | timm |
| MobileViT    | ★★★★☆ | ★★★☆☆ | 是 | 低 | [paper](https://arxiv.org/abs/2110.02178) [timm](https://github.com/huggingface/pytorch-image-models) | 否 | timm |
| Xception     | ★★☆☆☆ | ★★★★★ | 否 | 低 | [paper](https://arxiv.org/abs/1610.02357) [code](https://github.com/jfzhang95/pytorch-deeplab-xception) | 否 | 已集成 |
| ConvNeXt     | ★★★☆☆ | ★★★★★ | 是 | 低 | [paper](https://arxiv.org/abs/2201.03545) [timm](https://github.com/huggingface/pytorch-image-models) | 否 | timm |
| SwinTransf.  | ★★☆☆☆ | ★★★★★ | 是 | 中 | [paper](https://arxiv.org/abs/2103.14030) [code](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation) | 是 | timm/社区 |
| SegFormer    | ★★★★☆ | ★★★★★ | 否 | 中 | [paper](https://arxiv.org/abs/2105.15203) [code](https://github.com/NVlabs/SegFormer) | 是 | 轻量+高精度 |
| InternImage  | ★★☆☆☆ | ★★★★★ | 否 | 高 | [paper](https://arxiv.org/abs/2303.05631) [code](https://github.com/OpenGVLab/InternImage) | 是 | SOTA |

---

## 6. 参考资料与社区讨论

- [Papers With Code - Cityscapes Semantic Segmentation](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes)
- [知乎：Cityscapes 语义分割方法对比](https://zhuanlan.zhihu.com/p/349073282)
- [GitHub Issue 讨论（SegFormer）](https://github.com/NVlabs/SegFormer/issues?q=cityscapes)
- [mmsegmentation 主干适配文档](https://mmsegmentation.readthedocs.io/zh_CN/latest/)
- [timm 官方文档](https://huggingface.co/docs/timm/index)
- [Cityscapes 官方 benchmark](https://www.cityscapes-dataset.com/benchmarks/)

---

## 2025-06-14 进展与问题梳理

### 1. token_length 参数链路与 shape mismatch 问题
- 训练、评估时 token_length 需严格一致，否则权重加载 shape mismatch。
- 评估脚本已实现自动从权重或参数表读取 token_length，但 DeepLab 初始化参数传递存在错位，导致 token_length 仍为默认值。
- 临时将 DeepLab 默认 token_length 改为 120，评估可通过，根本问题待彻底修复。

### 2. 参数唯一来源方案
- 建议以参数表（experiment_records.csv/config.txt）为唯一参数源，训练、评估均从参数表读取，彻底消除歧义和 shape mismatch。
- 训练脚本、评估脚本均可自动同步参数表，归档权重与参数一一对应。

### 3. 代码链路 debug
- 已在 get_miou.py、deeplab.py、deeplabv3_reins.py 等关键处插入 debug 输出，便于全链路追踪 token_length。
- 发现 DeepLab __init__ 参数顺序与调用不一致，需全部用关键字参数传递。

### 4. 明日计划
- 彻底修复 DeepLab/DeeplabV3 参数链路，去除所有默认值，强制参数表唯一来源。
- 自动化参数同步、归档、复现流程。
- 代码推送到 pro6.14 分支。

如需某一主干的详细集成代码、权重下载脚本、实验配置文件、社区讨论精华等，请随时告诉我！
