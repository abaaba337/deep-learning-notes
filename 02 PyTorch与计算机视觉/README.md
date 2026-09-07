# 02 PyTorch与计算机视觉

[返回学习路径](../README.md)

按原笔记顺序学习 OpenCV、FNN/MNIST、CNN、RNN/LSTM、注意力、图像处理和强化学习概念。

- [CMIT Lecture Notes.ipynb](CMIT%20Lecture%20Notes.ipynb)

## 运行入口

在本目录启动 Jupyter，并打开 `CMIT Lecture Notes.ipynb`。笔记中的 `train`、`test`、`CNN` 等名称会被后续章节重新定义，请按章节顺序执行对应的导入和定义。

MNIST 数据保留了一份；也可由 `datasets.MNIST(download=True)` 获取。图像、视频、模型权重和股票示例的相对目录保留。

DRIVE 视网膜分割数据未随原资料提供；相关单元格需要先从 [DRIVE 官方网站](https://drive.grand-challenge.org/) 获取数据，按 `Notes_IMProcess/DRIVE/` 的训练/测试目录放置。缺失数据时不能完成该分割实验。模型加载示例需要先执行对应训练和保存单元格，不能把名称不同的旧权重视为兼容模型。

`Notes_RNN/StockPrediction243/GatedUnitsInRNN-main` 是保留的历史参考实现，其中训练/验证划分和自定义门控单元有局限；它不是本次测试覆盖的标准 LSTM 实现。优先阅读 Notebook 的 `nn.LSTM` 示例。
