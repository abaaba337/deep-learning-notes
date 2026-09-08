# 第 02 章　PyTorch 与计算机视觉

[打开 Notebook](notes.ipynb) | [返回学习路径](../README.md)

> **本章定位**：第 02 章聚焦张量、训练循环和图像任务；RNN、LSTM 与时序预测统一从第 03 章开始。

## 阅读顺序

- [2.1 PyTorch 与前馈神经网络](notes.ipynb#part-02-01)
- [2.2 OpenCV 与图像输入](notes.ipynb#part-02-02)
- [2.3 卷积神经网络](notes.ipynb#part-02-03)
- [2.4 U-Net 与语义分割](notes.ipynb#part-02-04)
- [2.5 图像去噪](notes.ipynb#part-02-05)

## 标题与运行约定

Notebook 使用一级章标题、二级 `章.节`、三级 `章.节.小节`、四级知识点；每节用分隔线、学习目标和小节导航突出边界。可从仓库根目录或本章目录启动，先执行路径与公共导入两格。FNN、ImageCNN、UNet 各定义一次；分割和去噪复用 ImageCNN，按节跳读时先运行其定义。不同任务的数据和训练函数使用独立名称。配图只存放在本章 `images/`。

## 数据与输出

- [视频输入](data/opencv/Ocean.mp4) 位于 data/opencv；MNIST 由 Notebook 的 download=True 获取。
- DRIVE 分割数据需从 [官方网站](https://drive.grand-challenge.org/) 获取并放入 `data/DRIVE/`，先训练和保存再执行模型加载示例。
- CNN 论文存放在 references。所有模型与图像结果写入根目录 `outputs/chapter-02/`，旧模型保留原文件名。U-Net 训练示例使用 8 起始通道，加载时必须采用相同通道配置。
- OpenCV 默认内嵌显示；桌面窗口用 SHOW_WINDOWS 显式开启。视频示例最多缓存 120 帧。
- 训练默认 1 epoch 用于检查流程。没有 DRIVE 时可完成结构前向演示并跳过真实数据训练；去噪沿用 MNIST，逐样本生成固定噪声，不预生成整套浮点数据。
- 分割展示的是概率图，评估为全图像素 micro Dice；未使用 DRIVE 的视野掩码，不能冒充官方 benchmark 分数。去噪展示首个测试批次的 MSE，不代表完整测试集。

[下一章：从 RNN/LSTM 开始序列建模](../03%20Transformer%E4%B8%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%85%A5%E9%97%A8/notes.ipynb#part-03-01)

## 轻量检查

在仓库根目录运行 `python scripts/check_vision.py`，直接提取本 Notebook 的定义检查形状、梯度、掩码、Dice、噪声与视频资源释放；依赖 PyTorch、NumPy 和 Pillow，不下载数据、不打开桌面窗口。
