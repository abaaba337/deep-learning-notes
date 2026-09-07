# 03 Transformer与大模型入门

[返回学习路径](../README.md)

从 token 与位置编码、注意力和编码器/解码器实现开始，再学习微调、压缩、训练稳定性、对齐、推理及评估。

- [LLM_Transformer.ipynb](LLM_Transformer.ipynb)

## 运行入口

在本目录打开 `LLM_Transformer.ipynb`，从顶部按顺序执行。前半部分演示模型结构和前向计算；它没有完整的语料训练流水线。后半部分是论文阅读笔记，空代码单元格表示尚未实现，不代表已有训练代码。

Generator 输出概率用于展示；使用 `CrossEntropyLoss` 训练时须传入 logits。该教学模型没有实现完整 padding 管理和高性能 KV cache。
