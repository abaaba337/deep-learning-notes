# 运行输出（本地，不提交）

所有新生成文件放在这里；Notebook 的内嵌输出在提交前清空。

- `chapter-02/`：教材保存的图像、FNN/CNN/U-Net/去噪权重。
- `chapter-03/`：时序预测的 LSTM 权重及股票参考实验输出。
- `vit-normalization/<run>/`：config.json、history.json、checkpoint.pt。
- `mllm-compression-safety/<run 或文件>/`：量化导出、模型回答、标注和指标。
- 两个实践的 `imported/`：原有训练曲线和模型回答，仅作为本地历史记录。

图表显示用 plt.show；需要保存时显式指向上述目录。配图源文件仍在各章 images，运行结果不能写回 images。大数据输入与输出区分，输入见 [datasets](../datasets/README.md)。这里除说明文件外均被 Git 忽略，可再运行生成。
