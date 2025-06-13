# 项目根目录说明

本目录包含项目主入口脚本、配置文件、依赖说明、文档等。

- README.md / README.txt：项目整体介绍与快速上手指南。
- requirements.txt：依赖包列表，使用 pip install -r requirements.txt 安装。
- config.py：全局路径与参数配置，供各脚本调用。
- train.py / predict.py / get_miou.py 等：训练、推理、评估主脚本。
- 学习记录.md / 常见问题汇总.md 等：学习笔记与经验总结。
- 其他 .py 文件：数据处理、辅助工具等脚本。

建议：主入口脚本统一放在 scripts/ 目录，工具脚本放 utils/，数据相关脚本放 data/scripts/，便于管理。
