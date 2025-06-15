import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 配置
CSV_PATH = 'results\\mious_deeplab\\miou_out_mobilenetv2rein_2025-06-10_23-46-32_eval_2025_06_14\\confusion_matrix.csv'
OUT_PATH = 'results\\mious_deeplab\\miou_out_mobilenetv2rein_2025-06-10_23-46-32_eval_2025_06_14\\confusion_matrix_heatmap.png'

# 读取混淆矩阵
cm = pd.read_csv(CSV_PATH, index_col=0)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
plt.show()

print(f'混淆矩阵热力图已保存到: {OUT_PATH}')
