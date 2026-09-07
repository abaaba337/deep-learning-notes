# 本地实践数据（不提交）

- `mm-safetybench/imgs/`：从原提供目录归并的图像数据，保留 scenario / SD、TYPO、SD_TYPO / ID.jpg 结构。重新获取方式见 [上游 MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench)；问题 JSON 在第 03 章实践 benchmark 中。
- `pile-val/val.jsonl.zst`：原 AWQ 校准语料，本地保留，未重复上传。正式入口接收 `calibration.jsonl`，可通过 datasets 的 JSON loader 读取原 zst，再固定种子选择文本后导出。
- `imagenet100/<class>/`：ViT 训练输入，原资料没有完整数据，用户自行准备同一 100 类子集；不要臆造旧类别映射。

第 02 章的少量输入视频/股票 CSV 位于该章 data；MNIST 可由 Notebook download=True 重新下载，因此清理了仓库内下载副本。DRIVE 仍需按章节说明自行获取。
