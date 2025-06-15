# Reins消融实验归档结构

本目录用于系统归档Reins相关消融实验的所有结果、参数和日志。

## 文件夹说明

- `single/`：仅主干输出与ASPP之间插入Reins（单点插入）
- `multi/`：多点插入Reins（如浅层、深层、ASPP后等）
- `no_reins/`：不使用Reins模块的baseline实验

每个子文件夹下建议按实验时间戳或参数命名新建子目录，归档config.txt、权重、日志、可视化等。

## 建议命名示例

```
results/reins_ablation/single/logs_20250614_120000/
results/reins_ablation/multi/logs_20250614_130000/
results/reins_ablation/no_reins/logs_20250614_140000/
```

## 归档内容建议
- config.txt（完整参数快照）
- 训练日志、loss曲线、权重文件
- 评估结果、可视化样例

---
如需自动化脚本批量生成实验目录、归档参数，请参考项目scripts或联系维护者。
