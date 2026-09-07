# Deep Learning Notes

深度学习与大模型学习笔记：从神经网络和 PyTorch 基础，逐步学习 Transformer、微调、训练与推理、RAG 和评估。每章入口见下表；01–03 的 Notebook 均命名为 `notes.ipynb`，02–03 按主题块衔接，正文内容保留。

## 学习路径

| 顺序 | 章节 | 学习内容 |
|---|---|---|
| 01 | [01 神经网络与反向传播](01%20%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD/README.md) | 先理解神经元、损失函数、梯度与链式法则，再进入代码。原 PDF 保留原样。 |
| 02 | [02 PyTorch与计算机视觉](02%20PyTorch%E4%B8%8E%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/README.md) | 按 FNN/MNIST → OpenCV → CNN → U-Net/分割 → 去噪 学习。 |
| 03 | [03 序列建模与 Transformer](03%20Transformer%E4%B8%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%85%A5%E9%97%A8/README.md) | 从 RNN/LSTM 与时序预测到注意力、Transformer 实现，再依次学习训练目标、稳定性、微调、压缩、强化学习、对齐与评估。 |
| 04 | [04 大模型专题与面试](04%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E4%B8%93%E9%A2%98%E4%B8%8E%E9%9D%A2%E8%AF%95/README.md) | 按需查阅 NLP 基础、模型架构、训练数据、分布式训练、微调、推理、强化学习、RAG、评估及应用。 |
| 05 | [05 数据结构与算法](05%20%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%E4%B8%8E%E7%AE%97%E6%B3%95/README.md) | 并行练习数组、二分查找、双指针、排序和滑动窗口，巩固编程基础。 |
| 06 | [06 环境与实践补充](06%20%E7%8E%AF%E5%A2%83%E4%B8%8E%E5%AE%9E%E8%B7%B5%E8%A1%A5%E5%85%85/README.md) | 按需查阅环境安装、PyTorch/CUDA、量化与原始补充材料；环境准备可在第 02 章之前完成。 |

主线按 01 → 02 → 03 → 04 阅读；05 是编程练习，06 是环境与实践补充。学完基础后，04 可按具体问题查阅。

## 使用方式

GitHub 可直接阅读 Markdown、PDF 和 Notebook。运行 Notebook 时，将工作目录设为对应章节目录，确保图片、数据和权重的相对路径可用。

先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装与系统、驱动兼容的 torch/torchvision，再安装学习依赖：

```bash
python -m pip install -r requirements.txt
jupyter lab
```

这是学习笔记合集，不是一键训练应用。OpenCV 弹窗需要图形桌面；部分章节需要下载数据或先训练模型。具体限制见章节 README。

## 维护与核验

```bash
python -m pip install -r requirements-check.txt
python scripts/check_notes.py
python scripts/check_examples.py
python scripts/check_practices.py
```

第一项检查 Markdown/Notebook 的本地链接和 Python 语法；第二项用 CPU 合成数据检查 Transformer 数学行为、因果掩码及 MNIST 网络的一步训练，不下载训练数据。

整理规则与来源见 [AGENTS.md](AGENTS.md)、[整理说明](整理说明.md) 和 [来源与致谢](SOURCES.md)。

## 实践路径与产物约定

学完 02 图像分类和 03.6 归一化后，进入第 03 章 `practices/vit-normalization`；学完 03.8 量化和 03.11 评估后，进入 `practices/mllm-compression-safety`。这两项实践的入口和练习在第 03 章 Notebook、README 中均可找到。

已训练权重见 [models](models/README.md)，输入大数据见 [datasets](datasets/README.md)，所有生成产物见 [outputs](outputs/README.md)。这些目录只提交说明和模型清单。Notebook 不提交可重跑的内嵌 output，依赖通过 requirements 安装，不提交安装包或虚拟环境。

02 章处理图像任务，03 章从序列建模开始；两章 Notebook 的节、小节均连续编号，并提供分节学习目标和导航。
