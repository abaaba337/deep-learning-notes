# Deep Learning Notes

深度学习与大模型学习笔记：从神经网络和 PyTorch 基础，逐步学习 Transformer、微调、训练与推理、RAG 和评估。每章入口见下表；01–03 的 Notebook 均命名为 `notes.ipynb`，02–03 按主题块衔接，正文内容保留。

## 学习路径

| 顺序 | 章节 | 学习内容 |
|---|---|---|
| 01 | [01 神经网络与反向传播](01%20%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD/README.md) | 先理解神经元、损失函数、梯度与链式法则，再进入代码。原 PDF 保留原样。 |
| 02 | [02 PyTorch与计算机视觉](02%20PyTorch%E4%B8%8E%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/README.md) | 按 FNN/MNIST → OpenCV → CNN → U-Net/分割 → 去噪 学习。 |
| 03 | [03 序列建模与 Transformer](03%20Transformer%E4%B8%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%85%A5%E9%97%A8/README.md) | 从 RNN/LSTM 与时序预测到注意力、Transformer 实现，再依次学习训练目标、稳定性、微调、压缩、强化学习、对齐与评估。 |
| 04 | [04 补充笔记](04%20补充笔记/README.md) | 概念问答、代码解析、大模型专题与面试、数据结构与算法，按需查阅。 |

主线按 01 → 02 → 03 阅读；第 04 章汇总补充资料，按问题查阅或并行练习。

<a id="environment"></a>

## 使用与环境

GitHub 可直接阅读 Markdown、PDF 和 Notebook。运行 Notebook 时，将工作目录设为对应章节目录，确保图片、数据和权重的相对路径可用。

建议在独立虚拟环境中安装，Notebook 内核必须使用同一个解释器。Windows 可先运行 `py -m venv .venv`，再激活 `.venv\Scripts\Activate.ps1`；Linux/WSL 可运行 `python3 -m venv .venv`，再运行 `source .venv/bin/activate`。基础练习可用 CPU；两个完整实践按各自 README 配置，历史 AutoAWQ 环境必须隔离。

先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装与系统、驱动兼容的 torch/torchvision，再安装学习依赖：

```bash
python -m pip install -r requirements.txt
jupyter lab
```

在实际 Notebook 内核执行以下检查，确认安装位置和 CUDA 运算：

```python
import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__, "build CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "cuDNN:", torch.backends.cudnn.version())
    x = torch.randn(16, 16, device="cuda")
    print("CUDA matmul:", (x @ x.T).mean().item())
```

安装排错按以下顺序进行：

1. 检查 `sys.executable` 和 `python -m pip --version`，避免终端与 Jupyter 使用不同环境；安装或升级后重启内核。
2. NVIDIA GPU 用 `nvidia-smi` 检查驱动。其 CUDA Version 表示驱动支持能力，不是当前 PyTorch 构建版本；后者看 `torch.version.cuda`。CPU 构建没有 CUDA 运行能力。
3. 按显卡架构、系统与驱动选官方实际发布的构建。通常二进制包已带运行所需 CUDA 依赖；编译自定义 CUDA 扩展才另核对 Toolkit。不要把旧机器的驱动记录当成所有机器的要求。
4. 运行 `python -m pip check`；遇到冲突优先新建隔离环境，避免同一环境混装不同来源的 torch。复现实验按 [历史版本表](https://docs.pytorch.org/get-started/previous-versions/) 匹配 torch/torchvision/torchaudio，后者只在音频任务需要时安装。不要拼接不存在的版本组合，例如 torch 2.5.1 与 cu128。

这是学习笔记合集，不是一键训练应用。OpenCV 弹窗需要图形桌面；部分章节需要下载数据或先训练模型。具体限制见章节 README。

## 维护与核验

```bash
python -m pip install -r requirements-check.txt
python scripts/check_notes.py
python scripts/check_examples.py
python scripts/check_vision.py
python scripts/check_practices.py
```

第一项检查 Markdown/Notebook 的本地链接和 Python 语法；第二项用 CPU 合成数据检查 Transformer 数学行为、因果掩码及 MNIST 网络的一步训练，不下载训练数据。第三项检查图像网络、掩码、去噪和视频资源释放；第四项检查 ViT 训练与保存、量化补丁和离线指标。

维护规则与整理约定统一见 [AGENTS.md](AGENTS.md)；来源与致谢见本页下方。

## 实践路径与产物约定

学完 02 图像分类和 03.6 归一化后，进入第 03 章 `practices/vit-normalization`；学完 03.8 量化和 03.11 评估后，进入 `practices/mllm-compression-safety`。这两项实践的入口和练习在第 03 章 Notebook、README 中均可找到。

已训练权重和所有生成产物统一见 [outputs](outputs/README.md)，输入大数据见 [datasets](datasets/README.md)。这些目录只提交说明和模型清单。Notebook 不提交可重跑的内嵌 output，依赖通过 requirements 安装，不提交安装包或虚拟环境。

02 章处理图像任务，03 章从序列建模开始；两章 Notebook 的节、小节均连续编号，并提供分节学习目标和导航。

<a id="downloads"></a>

## 数据集与模型下载

资源包：`deep-learning-notes-datasets-and-outputs.zip`

- [百度网盘下载](https://pan.baidu.com/s/1YckQ2Oq-yAFmDWMzbpTgDg)
- 提取码：`9sv3`

下载后，将资源包中的 `datasets/` 和 `outputs/` 放到本仓库根目录，与 README.md 和 01–04 章文件夹同级。若解压后多了一层同名文件夹，进入该文件夹再移动这两个目录，避免形成 `datasets/datasets/` 或 `outputs/outputs/`。

合并目录时保留仓库当前的 README 和 manifest，以及自己新生成的实验结果；不要直接覆盖整个 outputs。已有权重可按 [outputs/manifest.json](outputs/manifest.json) 中的路径、大小与 SHA-256 核对。资源包中的数据说明见 [datasets](datasets/README.md)，权重格式与输出用途见 [outputs](outputs/README.md)。未包含的数据或模型仍按对应实践说明准备。

## 来源与致谢

- 第 04 章含 [wdndev/llm_interview_note](https://github.com/wdndev/llm_interview_note) 的资料及原有个人整理，保留原作者署名、论文和文章链接。
- 各章节中的论文、课件、图片、示例和 Word 附件分别遵循原来源的使用条件。本仓库没有为第三方材料另行授予统一许可证。

### 技术参考

- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)：logits、LogSoftmax 与 NLLLoss。
- [PyTorch Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)：forward 调用机制、hooks 和梯度管理。
- [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html)：batch 数与 drop_last。
- [torchvision ToTensor](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html)：像素缩放条件。
- [PyTorch LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)：总体方差和 epsilon 在平方根内部。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：独立多头投影、注意力缩放与编码器/解码器掩码。
- [PyTorch 安装选择器](https://pytorch.org/get-started/locally/) 与 [历史版本](https://docs.pytorch.org/get-started/previous-versions/)：安装命令必须匹配实际发布版本。

### 实践来源

- ViT 基础实现源自 [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)，保留随附许可证和 MixLN 实验变体。
- 多模态量化使用 [AutoAWQ 0.2.7.post3](https://pypi.org/project/autoawq/0.2.7.post3/) 历史接口，必要兼容修复集中在实践 patch_autoawq.py，源码笔记见第 04 章 coding.md。
- 安全评估问题来自 [MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench)；调研链接和基准表集中在实践 README，独立 PDF 与许可证保留在实践目录。
