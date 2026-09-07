# 第 03 章　序列建模与 Transformer

[打开 Notebook](notes.ipynb) | [返回学习路径](../README.md)

> **本章定位**：把 RNN、LSTM、时序预测、注意力与 Transformer 放在同一条学习线上，再延伸到大模型训练、压缩、对齐和评估。

## 阅读顺序

- [3.1 序列建模：RNN、LSTM 与时序预测](notes.ipynb#part-03-01)
- [3.2 从序列建模到注意力](notes.ipynb#part-03-02)
- [3.3 Token、嵌入与位置编码](notes.ipynb#part-03-03)
- [3.4 编码器、解码器与掩码](notes.ipynb#part-03-04)
- [3.5 训练目标与 Teacher Forcing](notes.ipynb#part-03-05)
- [3.6 训练归一化与稳定性](notes.ipynb#part-03-06)
- [3.7 参数与内存高效微调](notes.ipynb#part-03-07)
- [3.8 模型压缩与剪枝](notes.ipynb#part-03-08)
- [3.9 强化学习基础](notes.ipynb#part-03-09)
- [3.10 AI 对齐](notes.ipynb#part-03-10)
- [3.11 推理与评估](notes.ipynb#part-03-11)

## 标题与运行约定

Notebook 使用一级章标题、二级 `章.节`、三级 `章.节.小节`、四级知识点；每节用分隔线、学习目标和小节导航突出边界。以本章为工作目录，先执行公共导入和输出路径配置，再按实验运行。同名变量会在不同主题中重定义。配图只存放在本章 `images/`。

## 时序数据与实践

- [股票输入数据](data/stock/stock.csv) 随 3.1 的时序预测示例保存。
- [历史门控参考代码](practices/stock-sequence/reference/README.md) 与教材同章；它有数据划分和自定义门控实现局限。教材中的股票示例也包含全序列标准化，不代表严格的样本外预测验证。
- RNN/LSTM 的新输出写入根目录 `outputs/chapter-03/`；已有时序模型归档位于 `models/chapter-03/`。

## Transformer 实现与配套项目

Transformer 部分提供结构、前向计算和数学验证，没有完整语料训练流水线。Generator 为展示输出概率，实际使用 CrossEntropyLoss 时应输入 logits。

- 3.6 后：[ViT 归一化与训练稳定性](practices/vit-normalization/README.md)。
- 3.8 和 3.11 后：[多模态模型压缩与安全评估](practices/mllm-compression-safety/README.md)。

两个项目各有 notes.ipynb、requirements 和独立运行入口；输出仍分别放在根目录 `outputs/vit-normalization/` 和 `outputs/mllm-compression-safety/`。GPU 重型实验按各自 README 准备。

[上一章：计算机视觉](../02%20PyTorch%E4%B8%8E%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/notes.ipynb) | [下一章：大模型专题](../04%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E4%B8%93%E9%A2%98%E4%B8%8E%E9%9D%A2%E8%AF%95/README.md)
