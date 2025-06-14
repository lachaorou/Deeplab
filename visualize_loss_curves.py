import pandas as pd
import matplotlib.pyplot as plt

# 读取合并后的 loss 和 val_loss
loss_df = pd.read_csv('merged_loss.csv')
val_loss_df = pd.read_csv('merged_val_loss.csv')

plt.figure(figsize=(10, 6))
plt.plot(loss_df['step'], loss_df['value'], label='Train Loss')
plt.plot(val_loss_df['step'], val_loss_df['value'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss Curve (断点续训自动拼接)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve.png')
plt.show()

print('已生成 loss_curve.png 并弹出可视化窗口')
