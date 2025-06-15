# Copilot启动助手脚本
# 每次启动时自动读取本项目的欢迎信息和实验状态，便于快速进入分割实验环境。

import os

WELCOME_FILE = "copilot_welcome.txt"

# 默认欢迎内容，可自定义
DEFAULT_WELCOME = """
欢迎使用 Cityscapes/VOC 语义分割自动化实验环境！

- 项目路径: {project_dir}
- 自动化排查与复现脚本清单: 自动加载
- 经验记录: 学习记录.txt
- 常用脚本: check_label_values.py, check_pred_unique.py, check_label_pred_pair.py, visualize_label_pred.py, get_miou.py

如需快速排查mIoU异常、类别空间、标签/预测一致性、可视化等，直接输入需求即可。

【提示】你可以随时关闭窗口，所有脚本和记录都已本地保存。
"""

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    welcome_path = os.path.join(project_dir, WELCOME_FILE)
    if not os.path.exists(welcome_path):
        with open(welcome_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_WELCOME.format(project_dir=project_dir))
    with open(welcome_path, "r", encoding="utf-8") as f:
        print(f.read())
