import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集路径
CITYSCAPES_ROOT = os.path.join(BASE_DIR, 'Cityscapes')
VOC_ROOT = os.path.join(BASE_DIR, 'VOCdevkit')
LABEL_DIR = os.path.join(VOC_ROOT, 'SegmentationClass')
IMAGE_DIR = os.path.join(VOC_ROOT, 'JPEGImages')
TRAIN_TXT = os.path.join(VOC_ROOT, 'ImageSets', 'Segmentation', 'train.txt')
VAL_TXT = os.path.join(VOC_ROOT, 'ImageSets', 'Segmentation', 'val.txt')

# 日志、权重、输出等
LOG_DIR = os.path.join(BASE_DIR, 'logs')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# 训练参数
BATCH_SIZE = 8
NUM_CLASSES = 19
EPOCHS = 100

# 其他参数
SEED = 42

class Config:
    # 路径相关
    save_dir = 'logs_mobilenetv2rein'
    VOCdevkit_path = 'VOCdevkit'
    # 训练参数
    Init_Epoch = 0
    Freeze_Epoch = 0
    Freeze_batch_size = 4
    UnFreeze_Epoch = 80
    Unfreeze_batch_size = 4
    Freeze_Train = False
    input_shape = [896, 640]
    downsample_factor = 8
    num_classes = 19
    backbone = 'mobilenet'
    pretrained = True
    model_path = ''
    # 优化器与学习率
    Init_lr = 7e-3
    Min_lr = 7e-3 * 0.01
    optimizer_type = 'sgd'
    momentum = 0.9
    weight_decay = 1e-4
    lr_decay_type = 'cos'
    # 其它训练参数
    save_period = 5
    eval_flag = True
    eval_period = 5
    dice_loss = False
    focal_loss = True
    # 类别权重（可自动统计，默认提供一组）
    cls_weights = [
        0.833209, 0.909937, 0.852506, 1.023590, 1.008330, 0.987683, 1.096596, 1.032917, 0.866557, 0.994650,
        0.929913, 0.989200, 1.120306, 0.903261, 1.078538, 1.082251, 1.093792, 1.150533, 1.046229
    ]
    num_workers = 8
    seed = 11
    Cuda = True
    distributed = False
    sync_bn = False
    fp16 = True
    use_tokens = True

# 兼容原有常量导入
config = Config()
