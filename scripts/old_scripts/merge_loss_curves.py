import pandas as pd
import glob

# 找到所有 loss/val_loss csv
loss_csvs = sorted(glob.glob('results/logs_deeplab/logs_mobilenetv2rein/loss_2025_06_*/events.out_loss.csv*'))
val_loss_csvs = sorted(glob.glob('results/logs_deeplab/logs_mobilenetv2rein/loss_2025_06_*/events.out_val_loss.csv*'))

# 合并 loss
loss_all = pd.concat([pd.read_csv(f) for f in loss_csvs], ignore_index=True)
loss_all['step'] = range(1, len(loss_all)+1)
loss_all.to_csv('merged_loss.csv', index=False)

# 合并 val_loss
val_loss_all = pd.concat([pd.read_csv(f) for f in val_loss_csvs], ignore_index=True)
val_loss_all['step'] = range(1, len(val_loss_all)+1)
val_loss_all.to_csv('merged_val_loss.csv', index=False)

print('已合并并导出 merged_loss.csv 和 merged_val_loss.csv')
