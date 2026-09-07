# 维护说明

- 根目录的 01–06 是学习路径；README 是主入口，各章 README 负责章内导航。
- 01–03 的主 Notebook 统一为 `notes.ipynb`；02–03 按当前主题块顺序衔接。用户要求调整布局时可移动完整主题块，保留块内正文与依赖顺序。
- Notebook 的实验名称会被后续章节复用，修改定义前必须检查所有调用单元格。
- 合并文件先比较内容；保留独有增补、图像、数据、出处和署名。不同角度的同主题笔记不等于重复文件。
- 每章配图只能存放在本章 `images/`，统一使用相对路径和正斜杠。删除前同时检查正文、HTML 与代码引用；训练数据、视频、模型权重不按配图删除。缺失外部数据应说明获取方式。
- Notebook 单元格 ID 必须保留；测试通过 ID 查找定义，不依赖会随布局变化的单元格序号。
- 执行 `python scripts/check_notes.py` 和 `python scripts/check_examples.py`；不以重型训练代替回归检查。
- 采用 ponytail full：复用标准库和原生接口，最小修改；完成代码修改后按 ponytail-review 检查 diff。
- 不提交虚拟环境、运行缓存、登录凭据或编辑器个人配置。保留第三方材料来源，不将第三方内容统一宣称为原创。

- `practices/` 下每个实践可有自己的 notes.ipynb。模型权重只放根目录 models；运行输出只写根目录 outputs；下载数据/安装包不提交。
- 提交前清空 Notebook outputs/execution_count，保留正文、代码和稳定 cell ID。代码注释中的结果不冒充实测。
- 新实践回归执行 `python scripts/check_practices.py`，不下载大模型，不默认发出 API 请求。

- 02 章只保留图像任务；RNN、LSTM、时序预测及配套数据/代码归入 03 章开头。
- 主 Notebook 标题统一为 H1 章、H2 章.节、H3 章.节.小节、H4 知识点；不得跳级。章节顺序调整后同步重编号、锚点、README 和配套实践引用。
