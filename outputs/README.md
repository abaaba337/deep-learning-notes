# 模型与运行输出（本地保存）

既有权重、新训练 checkpoint、量化导出、日志、图表、回答与指标统一存放在根目录 `outputs/`，按项目归类。这里不再区分 models 与 outputs，也不需要训练后再次搬运模型。输入数据仍在 [datasets](../datasets/README.md)，教材配图仍在各章 images。

| 子目录 | 内容 |
|---|---|
| `chapter-02/` | FNN、CNN、U-Net、去噪模型及生成图像；历史权重保留 Notes 开头的原文件名，新结果使用 fnn.pth、cnn.pth 等名称 |
| `chapter-03/` | LSTM 权重与历史股票模型；新股票实验保存到 `stock-reference/` |
| `chapter-04/` | 专题示例生成的训练 checkpoint、日志、模型转换结果与 PPO 权重 |
| `vit-normalization/<run>/` | 7 个历史 ViT/MixLN checkpoint，以及新运行的 config.json、history.json、checkpoint.pt；原训练曲线位于 imported/ |
| `mllm/<模型名>/` | 用户自行下载的完整浮点模型目录；目前没有随附 LLaVA 权重 |
| `mllm-compression-safety/` | 校准语料导出、量化模型、回答、标注与指标；原生成回答位于 imported/ |

## 权重清单与加载

已有 13 个模型文件的路径、字节数及 SHA-256 见 [manifest.json](manifest.json)，路径相对于仓库根目录。新克隆仓库只有说明与清单，权重和其他产物均不提交到 GitHub。**其中包含需要保留的既有训练权重，不要把整个 outputs 当作可随意清空的缓存。** 新运行使用独立名称，避免覆盖已有文件。

ViT 的 7 个 checkpoint 每个约 179 MiB，包含模型和优化器状态。先构建与运行名一致的 baseline/MixLN 结构，再用 `torch.load(path, map_location="cpu", weights_only=True)["model_state"]` 读取并 load_state_dict；它们不是通用 Transformers 格式。

第 02 章历史文件名与现有实验编号不同，不能将 CNN 权重当成 U-Net 权重。`Notes224FNN.pkl` 是依赖旧 `__main__.Net` 的完整对象，仅作归档；优先用相应 `.pth`，不加载来源不明的 pickle。教材默认先训练，再读取本次保存的结果。

## 使用与外部存储

01–03 Notebook 在对应章节目录运行；实践脚本的默认输出路径由脚本位置定位仓库根目录。第 04 章的可复制示例以仓库根目录作为工作目录。命令行指定输出时也选择 `outputs/<项目>/<运行名>`；图表需保存时写入对应子目录，不写回教材 images。Notebook 内嵌输出提交前清空。

向 Hugging Face 等外部存储上传时，挑选需要的权重并保留目录层级、架构/配置、来源和清单；不要把整个 outputs 中的临时结果一起上传。当前尚未上传，未提供下载地址。上传后在此记录地址与 revision，并同步 manifest；含优化器状态的文件不能宣称已转换为 Transformers 模型。

整理前代码与小文件的本地备份在项目同级 `DL 扫盲实践原始代码-20260908.zip`，该备份不包含大于 20 MB 的单文件；大权重现保存在本目录，输入数据在 datasets。
