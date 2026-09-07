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

Notebook 使用一级章标题、二级 `章.节`、三级 `章.节.小节`、四级知识点；每节用分隔线、学习目标和小节导航突出边界。以本章为工作目录，先执行公共导入和输出路径配置，再按实验运行。同名变量会在不同主题中重定义。配图只存放在本章 `images/`。

## 数据与输出

- [视频输入](data/opencv/Ocean.mp4) 位于 data/opencv；MNIST 由 Notebook 的 download=True 获取。
- DRIVE 分割数据需从 [官方网站](https://drive.grand-challenge.org/) 获取并放入 `data/DRIVE/`，先训练和保存再执行模型加载示例。
- CNN 论文存放在 references。新图像与模型输出写入根目录 `outputs/chapter-02/`；旧模型归档位于 `models/chapter-02/`。
- OpenCV 窗口示例需要桌面环境。

[下一章：从 RNN/LSTM 开始序列建模](../03%20Transformer%E4%B8%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%85%A5%E9%97%A8/notes.ipynb#part-03-01)
