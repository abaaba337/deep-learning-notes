# 02 PyTorch与计算机视觉

[打开 Notebook](notes.ipynb) | [返回学习路径](../README.md)

## 阅读顺序

- [2.1 PyTorch 与前馈神经网络](notes.ipynb#part-02-01)
- [2.2 OpenCV 与图像输入](notes.ipynb#part-02-02)
- [2.3 卷积神经网络](notes.ipynb#part-02-03)
- [2.4 U-Net 与语义分割](notes.ipynb#part-02-04)
- [2.5 图像去噪](notes.ipynb#part-02-05)
- [2.6 RNN、LSTM 与序列建模](notes.ipynb#part-02-06)

## 运行与资源

在本章节目录中启动 Jupyter，按主题执行导入、数据准备和模型定义。图片统一放在 `images/`，文件名相同时以内容摘要区分。数据、视频和模型权重不放入图片目录。

`train`、`test`、`CNN` 等名称会在不同实验中重定义，不要跨实验混用内核变量。图像实验完成后，RNN/LSTM 将学习重心从空间结构转向序列关系，再衔接第 03 章的注意力。

MNIST 下载副本已清理，通过 `datasets.MNIST(download=True)` 获取。OpenCV 弹窗需要桌面环境。

DRIVE 分割数据未随原资料提供，需从 [DRIVE 官方网站](https://drive.grand-challenge.org/) 获取并放到 `data/DRIVE/`；这是用户另行下载的实验数据，不是随笔记维护的配图。模型加载示例需先执行对应训练与保存步骤。

`practices/stock-sequence/reference` 保留历史参考实现，有数据划分和门控实现局限。Notebook 的股票例子也只是教学演示，不代表验证过的预测系统。

[上一章](../01%20%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD/notes.ipynb) | [下一章](../03%20Transformer%E4%B8%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%85%A5%E9%97%A8/notes.ipynb)

## 目录布局

| 目录 | 用途 |
|---|---|
| `images/` | 教材唯一配图目录 |
| `data/opencv/`、`data/stock/` | 输入视频和股票 CSV；重复 CSV 已合并 |
| `references/` | CNN 补充论文 |
| `practices/stock-sequence/reference/` | 历史手写门控参考代码，保留局限说明 |
| 根目录 `outputs/chapter-02/` | 运行后新生成的图像与权重 |
| 根目录 `models/chapter-02/` | 旧模型归档 |

旧 Notes_NN / Notes_IMProcess / Notes_OpenCV / Notes_RNN 目录已归并。第 02 章完成后继续第 03 章的 ViT 归一化实践；无需保留重复的安装包、压缩包和 MNIST 缓存。
