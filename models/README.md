# 本地模型仓库

训练好的模型统一放在本仓库根目录 `models/`，权重不提交到 GitHub。文件列表、字节数和 SHA-256 见 [manifest.json](manifest.json)。路径相对于学习仓库根目录；新克隆仓库只有清单，不包含这些权重。

| 本地子目录 | 内容 |
|---|---|
| `vit-normalization/<原运行名>/` | 7 个 ViT/MixLN checkpoint，每个约 179 MiB，含模型和优化器状态 |
| `chapter-02/` | 原 FNN、CNN 和去噪权重，保留原文件名以免误配 |
| `chapter-03/` | 已有 LSTM 与股票参考实验权重，随课程主题归类 |
| `mllm/<模型名>/` | 预留给用户自行下载的完整 LLaVA 模型目录；本次没有提供这些权重 |

ViT 加载用 `torch.load(path, map_location="cpu", weights_only=True)["model_state"]`，构建与运行名一致的 baseline/MixLN 结构后再 load_state_dict；原权重不是通用 Hugging Face Transformers checkpoint。02 章旧文件名和当前实验编号有差异，不能把历史 CNN 权重随意当作 UNet 权重；默认课程示例先训练，再从 `outputs/chapter-02/` 加载当前结果。

以后运行产生的 checkpoint、量化权重、曲线和回答统一写到根目录 `outputs/`。确认需要长期保留后，再挑选模型放进 `models/` 并更新清单。`models/` 存既有/选定模型，`outputs/` 存运行产物。

用户准备上传外部存储时，保留目录层级、manifest 与模型架构代码说明。若选择 Hugging Face，可将这些自定义 PyTorch checkpoint 作为文件托管，同时补充模型卡、运行配置和来源；目前尚未上传，也没有可用下载地址。上传完成后在此填写地址和 revision。不要把含优化器状态的 checkpoint 宣称为已转换的 Transformers 模型。

整理前的代码、研究文档和小文件另存于本地项目同级 `DL 扫盲实践原始代码-20260908.zip`。该备份不含大于 20 MB 的单文件；大权重保存在这里，数据在 `datasets/`。备份不进入 Git。

共保留 13 个模型文件，包含旧 FNN 的完整对象 pickle 与 state_dict 两种格式。`Notes224FNN.pkl` 依赖旧 `__main__.Net` 定义，仅作归档；运行优先使用相应 `.pth`，不要直接加载来源不明的 pickle。
