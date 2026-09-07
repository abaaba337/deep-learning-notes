# 03 Transformer与大模型入门

[打开 Notebook](notes.ipynb) | [返回学习路径](../README.md)

## 阅读顺序

- [3.1 从序列建模到注意力](notes.ipynb#part-03-01)
- [3.2 Token、嵌入与位置编码](notes.ipynb#part-03-02)
- [3.3 编码器、解码器与掩码](notes.ipynb#part-03-03)
- [3.4 训练目标与 Teacher Forcing](notes.ipynb#part-03-04)
- [3.5 训练归一化与稳定性](notes.ipynb#part-03-05)
- [3.6 参数与内存高效微调](notes.ipynb#part-03-06)
- [3.7 模型压缩与剪枝](notes.ipynb#part-03-07)
- [3.8 强化学习基础](notes.ipynb#part-03-08)
- [3.9 AI 对齐](notes.ipynb#part-03-09)
- [3.10 推理与评估](notes.ipynb#part-03-10)

## 运行与资源

在本章节目录中启动 Jupyter，按主题执行导入、数据准备和模型定义。图片统一放在 `images/`，文件名相同时以内容摘要区分。数据、视频和模型权重不放入图片目录。

先建立注意力直觉，再看 token 表示与完整编码器/解码器实现；随后依次学习训练目标、稳定性、微调、压缩、强化学习基础、对齐和评估。

代码部分提供结构与前向计算演示，没有完整语料训练流水线。Generator 为展示而输出概率，CrossEntropyLoss 训练应输入 logits。论文阅读部分不因布局调整而补成已实现的实验。

[上一章](../02%20PyTorch%E4%B8%8E%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/notes.ipynb) | [下一章](../04%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E4%B8%93%E9%A2%98%E4%B8%8E%E9%9D%A2%E8%AF%95/README.md)

## 配套实践

- 3.5 后：[ViT 归一化与训练稳定性](practices/vit-normalization/README.md)，把第 02 章图像分类和本章残差/归一化连接起来。
- 3.7 和 3.10 后：[多模态模型压缩与安全评估](practices/mllm-compression-safety/README.md)，依次完成 AWQ 校准、图文推理和评估分母检查。

两个实践各有 `notes.ipynb`、独立 requirements 和命令行入口；GPU 重型实验由用户在本地准备模型/数据后执行。运行输出统一到仓库根目录 outputs。
