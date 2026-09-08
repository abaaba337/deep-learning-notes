# 本地实践数据（不提交）

下载本目录的资源文件：[数据集与模型资源包（百度网盘）](../README.md#downloads)。提取码与解压步骤统一见该入口。

- `mm-safetybench/imgs/`：从原提供目录归并的图像数据，保留 scenario / SD、TYPO、SD_TYPO / ID.jpg 结构。重新获取方式见 [上游 MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench)；问题 JSON 在第 03 章实践 benchmark 中。
- `pile-val/val.jsonl.zst`：原 AWQ 校准语料，通过上述资源包获取。正式入口接收 `calibration.jsonl`，可通过 datasets 的 JSON loader 读取原 zst，再固定种子选择文本后导出。
- `imagenet100/<class>/`：ViT 训练输入，原资料没有完整数据，用户自行准备同一 100 类子集；不要臆造旧类别映射。

第 02 章的输入视频位于该章 data/opencv；股票 CSV 位于第 03 章 data/stock；MNIST 可由 Notebook download=True 重新下载，因此清理了仓库内下载副本。DRIVE 仍需按章节说明自行获取。
