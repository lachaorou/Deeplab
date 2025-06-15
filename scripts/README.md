# Scripts Directory

This directory contains all main entry-point scripts for training, evaluation, and prediction.

## Usage
- Run scripts here to train models, evaluate mIoU, predict on new data, or perform batch operations.
- Example: `train.py` is the main training script; `get_miou.py` is for evaluation; `predict.py` is for inference.

## Script Relationships
- These scripts depend on modules from Models, Utils, Dataset, and may call Automation or Inspect scripts for data preparation and checking.
- If you change dataset structure, model definitions, or utility functions, update the relevant paths and imports in these scripts.

# scripts 目录脚本用途索引

本目录包含训练、推理、评估、批量实验、可视化、数据处理等各类主流程与辅助脚本。以下为每个脚本的简要说明，便于查阅和协作：

- **train.py**：主训练入口，支持参数交互、权重归档、断点续训。
- **train_template.py**：通用训练模板，便于自定义和扩展不同模型结构。
- **predict.py**：推理/预测主入口，支持单张、批量、视频等多种模式。
- **get_miou.py**：模型评估主入口，自动计算mIoU、mPA等分割指标。
- **run_reins_ablation.py**：Reins结构消融实验主控脚本，自动批量对比不同结构。
- **run_multi_reins_experiments.py**：多组消融实验批量运行脚本，支持参数归档。
- **auto_resume_train.py**：自动断点续训脚本，检测最新权重恢复训练。
- **auto_split_train_val.py**：自动划分训练/验证集，支持自定义比例。
- **auto_colorize_detection_results.py**：批量上色检测结果，便于可视化分析。
- **check_label_validity.py**：批量检查标签有效性，输出无效标签文件。
- **check_pred_files.py**：检查预测结果文件的完整性和命名规范。
- **clean_val_txt.py**：清理验证集txt文件中的无效或重复项。
- **dataloader.py**：数据加载器示例或测试脚本。
- **export_visualization_to_pdf.py**：将可视化结果导出为PDF报告。
- **export_visualization_to_word.py**：将可视化结果导出为Word报告。
- **extract_tfevents.py**：提取TensorBoard日志事件，便于分析。
- **generate_loss_weights.py**：自动生成类别损失权重，支持自定义类别分布。
- **merge_loss_curves.py**：合并多组实验的loss曲线，便于对比。
- **plot_confusion_matrix.py**：混淆矩阵可视化脚本。
- **select_val_cover_all_classes.py**：自动选择覆盖所有类别的验证集样本。
- **stat_label_distribution.py**：统计标签类别分布，辅助权重生成。
- **visualize_loss_curves.py**：loss曲线可视化脚本。
- **visualize_samples.py**：批量可视化样本输入、标签、预测结果。
- **visualize_saved_checkpoints.py**：可视化已保存权重的模型表现。

---
如有脚本功能重叠或不常用，可后续合并、精简，保持目录简洁。建议主入口脚本（如train.py、predict.py、get_miou.py）重点维护，其余辅助脚本可根据实际需求优化。
