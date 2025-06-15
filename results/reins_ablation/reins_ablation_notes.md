# Reins 模块消融实验笔记

## 1. 实验目的
系统分析Reins模块在不同插入点、参数配置下对分割模型性能的影响，验证其有效性与适用性。

## 2. 实验设计
- **实验类型**：
  - 单点插入（single）：如backbone末端、ASPP前等
  - 多点插入（multi）：如backbone多层、ASPP前后等
  - 无Reins（no_reins）：作为基线对照
- **参数归档**：每次实验均归档config.txt、日志、权重等，便于复现与对比
- **自动化**：采用run_reins_ablation.py批量运行与归档

## 3. 主要参数
- reins_pos：Reins插入位置（如aspp、backbone、aspp+backbone等）
- reins_channels：Reins模块通道数
- 其它训练参数：batch_size、学习率、backbone类型等

## 4. 结果归档规范
- 每次实验自动生成独立子目录，归档config.txt、日志、模型权重、可视化结果等
- 结果目录结构：
  - results/reins_ablation/single/...
  - results/reins_ablation/multi/...
  - results/reins_ablation/no_reins/...

## 5. 典型发现与分析（示例）
- 单点插入Reins于ASPP前，mIoU提升明显，参数量增加有限
- 多点插入可进一步提升表现，但需注意显存与收敛速度
- 无Reins为基线，便于量化模块贡献

## 6. 建议与后续
- 持续补充不同插入点、参数组合的实验
- 可扩展自动化对比、差异报告生成
- 建议团队成员统一归档与命名规范，便于协作与复现

---
如有新实验或分析，请在本文件持续补充。
