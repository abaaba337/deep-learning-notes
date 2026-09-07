# 实践 1：ViT 归一化与训练稳定性

[返回第 03 章](../../README.md) · [练习 Notebook](notes.ipynb) · [模型清单](../../../models/README.md)

从第 02 章的图像分类进入 ViT：图像切成 patch → patch embedding 与位置编码 → Transformer → 分类 logits。在第 03 章 3.6 学完 Pre-LN、Post-LN 后，比较 baseline、PostPre、PreB2TPost 和 PrePost。实际实现是标准多头自注意力，旧 Notebook 标题中的 Linformer 不适用。

## 代码与学习顺序

1. 阅读 [utils/vit.py](utils/vit.py)：224×224 图像、32×32 patch 得到 49 个 patch，加上 CLS 后是 50 个 token；dim=392、12 层、8 头，每头 49 维，100 类。
2. 阅读 [utils/mixln_vit.py](utils/mixln_vit.py)：`Block.forward` 定义三种残差连接；`MixTypeLNTransformer.forward` 决定层的先后顺序，alpha 是 Post/B2TPost 层占比，层数取 floor(alpha × depth)。
3. 阅读 [train.py](train.py)：ImageFolder 固定类别映射，分层划分训练/验证集；训练开启 dropout 和梯度，验证关闭它们，指标按样本数累计，学习率调度使用实际 batch 总数。
4. 固定数据划分、种子、学习率和训练步数后比较曲线；记录显存峰值及耗时。历史 baseline 有末端 LayerNorm，MixLN 没有，因此旧实验不是只改变层顺序的严格单变量消融。为兼容权重，这里保留原架构。

## 环境与运行

CPU 可运行练习及小型回归；完整 ImageNet100 训练需要 GPU。建议 Python 3.11/3.11 独立环境，先安装匹配的 torch/torchvision，再 `python -m pip install -r requirements.txt`。不把现有环境打包上传。Windows 默认 `--workers 0`，避免 Notebook 多进程入口问题。

在本实践目录运行：

```bash
python train.py --help
python train.py --data ../../../datasets/imagenet100 --variant baseline --epochs 200 --batch-size 32 --lr 0.03 --output ../../../outputs/vit-normalization/baseline-lr003
python train.py --data ../../../datasets/imagenet100 --variant PostPre --alpha 0.25 --epochs 200 --batch-size 32 --lr 0.03 --output ../../../outputs/vit-normalization/postpre-a025-lr003
```

数据需为 `datasets/imagenet100/<class>/*.JPEG`；随附资料没有完整 ImageNet100 数据及历史类别映射。不要用重新排序的类别映射解释旧 checkpoint 的语义标签。原设置 batch=512，这里默认 32 便于启动，不声称复现原曲线；显存依批量和设备实测，内存不足先减 batch。

输出包含 `config.json`（类别映射、验证文件列表、超参数）、`history.json`、`checkpoint.pt`。新运行必须选择空目录，避免覆盖历史结果。新检查点保留优化器、调度器、epoch；当前入口从头训练，没有假称支持精确断点续训。

## 已有实验与复现限制

7 个原 checkpoint 已移动到根目录 `models/vit-normalization/`，原训练曲线位于 `outputs/vit-normalization/imported/<run>/history.json`。它们只在本地保存，不进 Git。原分析 Notebook 引用了一个未提供的 `preB2Tpost_lr3e-2_alpha50` checkpoint，不能据此补造第 8 次实验。

旧曲线来自未正确切换 eval 的训练代码；应视为历史记录，而非修正后结果。checkpoint 中缺少类别映射及完整运行元数据；下面的练习验证结构可加载，不把旧结果描述成完整复现。

来源：[lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)，保留随附 MIT 许可证（如原目录提供）。MixLN 实验代码与历史研究笔记一起保留。完整备份位置见根目录模型说明。
