# Cityscapes官方19类配色（顺序严格对应类别0~18）
CITYSCAPES_COLORMAP = [
    (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153),
    (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
    ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
    (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32)
]

def colorize_mask(mask, colormap=CITYSCAPES_COLORMAP):
    """
    将单通道mask按照Cityscapes官方配色上色，255为ignore，设为黑色。
    """
    import numpy as np
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(colormap):
        color_mask[mask == label] = color
    color_mask[mask == 255] = (0, 0, 0)
    return color_mask
