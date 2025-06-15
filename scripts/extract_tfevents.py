import os
from tensorboard.backend.event_processing import event_accumulator
import glob
import csv

# 递归查找所有 tfevents 文件
tfevent_files = glob.glob('results/logs_deeplab/logs_mobilenetv2rein/**/events.out.tfevents.*', recursive=True)

for tfevent in tfevent_files:
    print(f"\n==== 解析: {tfevent} ====")
    ea = event_accumulator.EventAccumulator(tfevent)
    ea.Reload()
    tags = ea.Tags()
    print("可用标量:", tags.get('scalars', []))
    # 你可以根据实际情况调整要提取的标量名
    for scalar_name in ['loss', 'val_loss', 'miou', 'epoch_miou']:
        if scalar_name in tags.get('scalars', []):
            values = ea.Scalars(scalar_name)
            print(f"\n[{scalar_name}]")
            for v in values:
                print(f"step={v.step}, value={v.value}")
            # 可选：导出为csv
            csv_path = tfevent.replace('.tfevents.', f'_{scalar_name}.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'value'])
                for v in values:
                    writer.writerow([v.step, v.value])
            print(f"已导出: {csv_path}")
        else:
            print(f"[{scalar_name}] 不存在于该日志")

print("\n全部解析完成。")
