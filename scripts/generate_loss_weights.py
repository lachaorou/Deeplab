# 生成loss加权权重数组脚本
# 直接运行即可输出可用于CrossEntropyLoss等的class_weight参数

# 统计结果（类别占比，按stat_label_distribution.py输出顺序填写）
ratios = [
    0.369281, 0.060063, 0.226312, 0.006266, 0.008397, 0.012601, 0.002078, 0.005657, 0.162870, 0.011216,
    0.038925, 0.012446, 0.001506, 0.069898, 0.002730, 0.002351, 0.001864, 0.000815, 0.004723
]

import numpy as np

# 计算权重（1/占比），并归一化
weights = np.array([1/r if r > 0 else 0 for r in ratios])
weights = weights / weights.max()  # 归一化到[0,1]

print('类别权重数组（可用于loss class_weight参数）:')
print(weights.tolist())

# 如需保存为txt或npy文件，可取消下方注释
# np.savetxt('class_weights.txt', weights)
# np.save('class_weights.npy', weights)
