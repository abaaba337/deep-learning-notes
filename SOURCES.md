# 来源与致谢

- CMIT 指 Centre for Mathematical Imaging Techniques；课程笔记保留其原始课程名称及正文中的引用。
- 第 04 章含 [wdndev/llm_interview_note](https://github.com/wdndev/llm_interview_note) 的资料及原有个人整理，保留原作者署名、论文和文章链接。
- 各章节中的论文、课件、图片、示例和 Word 附件分别遵循原来源的使用条件。本仓库没有为第三方材料另行授予统一许可证。

## 本次技术纠正依据

- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)：logits、LogSoftmax 与 NLLLoss。
- [PyTorch Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)：forward 调用机制、hooks 和梯度管理。
- [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html)：batch 数与 drop_last。
- [torchvision ToTensor](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html)：像素缩放条件。
- [PyTorch LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)：总体方差和 epsilon 在平方根内部。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：独立多头投影、注意力缩放与编码器/解码器掩码。
- [PyTorch 安装选择器](https://pytorch.org/get-started/locally/) 与 [历史版本](https://docs.pytorch.org/get-started/previous-versions/)：安装命令必须匹配实际发布版本。

## 实践项目

- ViT 基础实现源自 [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)，保留随附许可证和 MixLN 实验变体。
- 多模态量化使用 [AutoAWQ 0.2.7.post3](https://pypi.org/project/autoawq/0.2.7.post3/) 历史接口，自定义改动以差异文件保存。
- 安全评估问题来自 [MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench)；原调研 Word / PDF / Excel 保留在实践 references。
