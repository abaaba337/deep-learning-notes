# 维护与整理约定

## 文档职责与学习路径

- 根 README 是学习入口，集中维护通用环境、排错、来源与致谢；项目专用依赖放对应实践 README，不复制到补充笔记。
- 本文件维护结构规则、合并原则和验证边界；不另建 SOURCES 或整理流水账。CLAUDE.md 通过 `@AGENTS.md` 引用本文件。
- 01–06 构成学习路径：01 神经网络基础，02 PyTorch 与图像任务，03 序列建模与 Transformer，04 专题与面试，05 算法，06 补充笔记。
- 02 按 FNN → OpenCV → CNN → U-Net → 去噪；03 从 RNN/LSTM/时序开始，再到注意力、Transformer、训练目标、稳定性、微调、压缩、强化学习、对齐和评估。
- 01–05 的章 README 提供章内导航；06 只保留 concepts.md 和 coding.md，分别承担概念问答与代码解析，二者互链。

## 内容、编号与去重

- 01–03 主 Notebook 统一命名 notes.ipynb；实践也可有自己的 notes.ipynb。允许移动完整主题块，保留块内正文、公式和依赖顺序。
- 主 Notebook 使用 H1 章、H2 章.节、H3 章.节.小节、H4 知识点，不跳级；调整后同步编号、锚点、目录与实践引用。
- 保留稳定 cell ID；测试依 ID 查找定义。实验名可能被后续单元格复用，修改前搜索所有调用方。
- 同主题不等于重复：先比较正文、表格、公式、图片和代码，合并相同内容并保留独有增补、出处、署名。删除原件前确认转换完整且有可恢复副本。
- 每章只用一个 images/ 存教材配图，引用用正斜杠相对路径；删除前检查 Markdown、HTML、Notebook 和代码引用。训练图像、视频和模型权重不是教材配图。
- 重复安装包源码不留第二套实现。AutoAWQ 的必要修复集中于 patch_autoawq.py，历史学习注释经纠正并入 coding.md；第三方许可证保留。

## 运行数据与提交范围

- 输入大数据放根 datasets，既有权重、训练 checkpoint、日志、图表和预测统一放根 outputs，按章节/实践分子目录；只提交说明及模型清单，不提交可重跑产物。
- 模型移动前后校验 SHA-256，同步 outputs/manifest.json；数据压缩包删除前验证解压内容。不要提交安装包、虚拟环境、缓存、凭据或个人编辑器配置。
- Notebook 提交前清空 outputs/execution_count，保留代码、正文、cell ID。权重与数据仍在本地，不代表其他机器已具备运行条件。
- 外部数据缺失时写清获取方式，不能伪造数据或下载完成状态。DRIVE 分割需另备数据与匹配权重；历史时序示例需注意时间划分和信息泄漏。

## 技术纠正与验证

- 保持已纠正口径：MSE/MAE 的平均维度、反传与更新的区别、CrossEntropyLoss 接收 logits、ToTensor 缩放条件、DataLoader 的 drop_last、Module hooks、多头独立投影、LayerNorm 方差与 epsilon、因果掩码，以及 batch/device/NumPy 转换约束。精确依据统一链接根 README 或对应补充笔记。
- PEFT 不保证全量微调效果；LoRA 表示低秩更新，GaLore 投影梯度；state_dict 包含冻结参数和持久 buffer；AWQ clipping 是校准重构误差搜索。环境命令依真实发布组合维护，不沿用旧硬件配置推断兼容性。
- 采用 ponytail full：先读真实调用链，依次考虑删除需求、仓内复用、标准库、平台能力、现有依赖与最小实现。共享问题在根因处修复；有代码改动完成 ponytail-review，不引入无调用方抽象。
- 运行 `python scripts/check_notes.py` 检查链接、语法、配图与标题；运行 `python scripts/check_examples.py` 检查教材数学和基础训练；运行 `python scripts/check_practices.py` 检查实践入口与核心行为。
- 上述 CPU 合成检查不下载数据、大模型或调用付费 API。已有模型加载检查、轻量训练步骤不等于完整 GPU 训练、量化或安全评测；记录真实验证范围，注释结果不冒充实测。
- 安全指标把未标注和无效标签计为 unjudged；先报告覆盖率，不把原生成回答直接视为已完成评测。具体标注规则只在实践 README 维护。
