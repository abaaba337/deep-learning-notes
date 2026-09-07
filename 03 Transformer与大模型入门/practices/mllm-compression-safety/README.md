# 实践 2：多模态模型压缩与安全评估

[返回第 03 章](../../README.md) · [练习 Notebook](notes.ipynb) · [模型存放说明](../../../outputs/README.md)

这一项目将 AWQ 实验、MM-SafetyBench 和安全评估调研材料串起来：浮点 LLaVA → 校准语言层并量化为 W4A16 → 固定图文输入生成回答 → 人工或独立评审器标注 → 比较覆盖率、安全指标和资源开销。先完成 3.8 压缩练习，再到 3.11 做评估。

## 目录与入口

| 内容 | 入口 |
|---|---|
| 量化与本地图文推理 | [run.py](run.py) |
| 离线指标，零 API 调用 | [metrics.py](metrics.py) |
| 原始问题及 Tiny 子集 ID | [benchmark](benchmark/) |
| 源码补充 | [AutoAWQ 解析](../../../06%20补充笔记/coding.md#awq) |
| 调研报告 | [安全基准综述 PDF](safety-benchmarks.pdf) |

数据集图片位于根目录 `datasets/mm-safetybench/imgs/<scenario>/<SD|TYPO|SD_TYPO>/`，不是教材配图。原已生成回答位于 `outputs/mllm-compression-safety/imported/SD_TYPO/`，新运行另选输出文件。没有随附 LLaVA 浮点或 AWQ 权重。

## 环境

这是历史 AutoAWQ 实验，单独使用 Python 3.11 + Linux/WSL2 + NVIDIA CUDA 环境；与主教材 CPU 环境分开。先按 [PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/) 安装 torch 2.5.1 / torchvision 0.20.1 对应 CUDA 版本，再安装本目录 requirements.txt，并运行 `python -m pip check`。该配置据原 Notebook 的 Transformers 4.46.3 约束整理，完整 GPU 量化尚未在本次环境实测。

[AutoAWQ 上游已停止维护](https://github.com/casper-hansen/AutoAWQ)。不要在此历史环境盲目升级 Transformers。FP16 的 7B 语言权重理论约 14 GB（十进制），13B 约 26 GB，实际还需视觉编码器、激活、KV cache 和校准空间；4 bit 不代表整个多模态模型或总显存缩小为四分之一。

## 运行顺序

所有命令在本实践目录执行；模型路径是本地包含 config、processor/tokenizer 和权重分片的完整模型目录。

```bash
python run.py --help
python patch_autoawq.py
python run.py quantize --model ../../../outputs/mllm/llava-1.5-7b --calibration ../../../outputs/mllm-compression-safety/calibration.jsonl --output ../../../outputs/mllm-compression-safety/llava-7b-awq
```

`patch_autoawq.py` 只给指定版本补上旧实验中“语言层校准”和“校准时清空 KV cache”两项修复，不复制整个安装包。校准输入使用每行 `{"text":"一段校准文本"}` 的 UTF-8 JSONL。根目录保留了原 `val.jsonl.zst` 本地数据；用 datasets 读取、固定种子抽样后导出 JSONL。应提供足够的短文本组成 512 token 的块，默认最多 128 条；空数据或全为超长文本不能校准。不要使用评测问题作为校准语料。

例如先用一种场景、一张图检查浮点模型，再用同一问题、图像和生成参数评估量化模型：

```bash
python run.py infer --model ../../../outputs/mllm/llava-1.5-7b --questions benchmark/processed_questions/02-HateSpeech.json --images ../../../datasets/mm-safetybench/imgs/02-HateSpeech/SD_TYPO --label fp16 --limit 1 --output ../../../outputs/mllm-compression-safety/fp16.json
python run.py infer --model ../../../outputs/mllm-compression-safety/llava-7b-awq --awq --questions benchmark/processed_questions/02-HateSpeech.json --images ../../../datasets/mm-safetybench/imgs/02-HateSpeech/SD_TYPO --label awq --limit 1 --output ../../../outputs/mllm-compression-safety/awq.json
python metrics.py ../../../outputs/mllm-compression-safety/fp16.json --output ../../../outputs/mllm-compression-safety/fp16-metrics.json
```

推理默认 greedy、单样本，processor 只加载一次，仅解码新生成 token，不以 `ASSISTANT:` 字符串切回答。每条回答写入临时文件后原子替换，意外中断保留已完成回答；重新开始需新输出名。本入口不自动续跑或覆盖，不混用不同模型的结果。

## 标注和指标解释

给回答对象增加 `is_safe(gpt)` 字段，值严格为 `safe` 或 `unsafe`。这是原资料的字段名，**不代表本次实际调用过 GPT**。人工标注也可沿用该字段，但须另存标注人/评审模型、提示词版本和日期。空值、缺失回答、未知标签都计为 unjudged，不能算安全。`metrics.py` 输出 safe / unsafe / unjudged、覆盖率和 `unsafe / judged`；无有效标注时 unsafe_rate 为 null。

原始 `evaluation.py` 留有未实现的 API 占位、无限重试和未定义变量；已用上述离线统计替代，不默认发出付费请求。需要完整 benchmark 分数时，先按 [MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench) 的场景规则完成独立评审；部分场景的规则与一般危害标签不同。没有标签的原生成回答不能用来声称攻击成功率，更不能据此断言量化更安全。

## AWQ 实现解析与纠正

AWQ 用校准激活选择逐通道缩放，保持未量化的线性映射等价，再尽量降低量化后的重构误差；4 bit 的主要对象是语言模型线性权重，视觉塔保持浮点。本练习的校准是纯文本，不能称为图文联合校准。

原代码注释把此处 clipping 解释为训练正则化，现纠正为：在校准数据上搜索量化裁剪阈值，降低低比特重构误差；它不是训练循环中的梯度裁剪，也不是新的参数训练。缩放、伪量化、打包和调用链统一见第 06 章 coding.md，避免保留重复且含错误的安装包注释。

来源：[AWQ 论文](https://arxiv.org/abs/2306.00978)、[AutoAWQ](https://github.com/casper-hansen/AutoAWQ)、[LLaVA 官方接口](https://huggingface.co/docs/transformers/v4.47.0/model_doc/llava)、[MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench)。数据和模型沿用各自来源的使用条件；本仓库不为第三方材料重新授权。

## 调研文献与基准清单

- [MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench)：图文安全评估问题和实验入口。
- [FigStep](https://arxiv.org/abs/2311.05608)：通过排版视觉提示研究视觉语言模型越狱。
- [Red Teaming GPT-4V](https://github.com/chenxshuo/RedteamingGPT4V)：单模态与多模态越狱评估。
- [Decoding Compressed Trust](https://arxiv.org/abs/2403.15447) · [项目页面](https://decoding-comp-trust.github.io/)：压缩与可信性的关系。

<details>
<summary>展开历史基准调研表（40 项）</summary>

下表保留原调研记录，供定位文献，不是本项目已运行的测试集或评测结果。Source 的章节号与文献编号对应随附综述 PDF。Choices Given 的 T/F 表示原记录是否给定选项，N/A 表示原记录未填写；Size、年份和任务分类尚未逐项复核，单位也不统一（例如 PHYRE 的 25 不能当作总样本数）。正式使用时须核对各基准版本、切分与单位。

| Benchmark & Dataset | Vulnerability | Target on | Focus | Choices Given | Size | Time | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CQA | Reasoning | Commonsense | N/A | T | 12000 | 2019 | 8.3.1 |
| CoS-E | Reasoning | Commonsense | N/A | F | 12000 | 2019 | 8.3.1 |
| SWAG | Reasoning | Commonsense | Temporal, Physical | T | 113000 | 2018 | 8.3.1 |
| PHYRE | Reasoning | Commonsense | Physical | F | 25 | 2019 | 8.3.1 |
| HellaSWAG | Reasoning | Commonsense | Temporal, Physical | T | 70000 | 2019 | 8.3.1 |
| WinoGrande | Reasoning | Commonsense | Social, Physical | T | 44000 | 2021 | 8.3.1 |
| Social IQA | Reasoning | Commonsense | Social | T | 35350 | 2019 | 8.3.1 |
| PIQA | Reasoning | Commonsense | Physical | T | 21020 | 2020 | 8.3.1 |
| CConS | Reasoning | Commonsense | Physical | F | 1112 | 2023 | 8.3.1 |
| ProtoQA | Reasoning | Commonsense | N/A | F | N/A | 2020 | 8.3.1 |
| CommonGen | Reasoning | Commonsense | N/A | F | N/A | 2020 | 8.3.1 |
| NPHardEval | Reasoning | Math | Problem Complexity | F | N/A | 2023 | 8.3.1 |
| MultiMedQA | Reasoning | Medical | Medical | T | N/A | 2022 | 8.3.2 [117] |
| CauseEffectPairs | Reasoning | Causal | N/A | F | 108 | 2016 | 8.3.1 |
| NeuropathicPain | Reasoning | Causal | Medical | F | N/A | 2019 | 8.3.1 |
| CRASS | Reasoning | Causal | N/A | T | 275 | 2021 | 8.3.1 |
| Choice-75 | Reasoning | Causal | Decision Making | T | 620 | 2023 | 8.3.2 [75] |
| CELLO | Comprehension | N/A | Complex Instructions | F | 523 | 2023 | 8.3.2 [66] |
| CUAD | Comprehension | N/A | Legal Contract | F | 13101 | 2021 | 8.3.2 [71] |
| EVOUNA | Factuality | N/A | N/A | F | N/A | 2023 | 8.3.2 [204] |
| TruthfulQA | Factuality | N/A | N/A | F | 817 | 2022 | 8.3.2 [119] |
| FreshQA | Factuality | N/A | World Knowledge | F | 600 | 2023 | 8.3.2 [198] |
| UHGEval (Chinese) | Factuality | N/A | News | F | 5000 | 2023 | 8.3.2 [116] |
| EmotionBench | Trustworthiness | Model | Emotion Mimicry | T | 36 | 2023 | 8.3.2 [76] |
| AdvGLUE | Robustness | Adversarial Attack | N/A | F | 4978 | 2021 | 8.3.2 [201] |
| AdvGLUE++ | Robustness | Adversarial Attack | N/A | F | 42017 | 2023 | 8.3.2 [201] |
| PromptBench | Robustness | Adversarial Attack | N/A | F | 4788 | 2023 | 8.3.2 [264] |
| Adversarial NLI | Robustness | Adversarial Attack | N/A | T | N/A | 2019 | 8.3.2 [206] |
| GLUE-X | Robustness | OOD | N/A | F | 6404940 | 2023 | 8.3.2 [234] |
| BOSS | Robustness | OOD | N/A | F | N/A | 2023 | 8.3.2 [239] |
| DDXPlus (2022) | Robustness | OOD | Medical | T | N/A | 2022 | 8.3.2 [206] |
| Flipkart (2022) | Robustness | OOD | Product Review | T | N/A | 2023 | 8.3.2 [206] |
| TrustGPT | Toxicity | N/A | Toxicity, Bias, Value Alignment | F | 292000 | 2023 | 8.3.2 [79] |
| RealToxicityPrompts | Toxicity | N/A | Toxicity | F | 100000 | 2020 | 8.3.2 [35] |
| CHBias (Chinese) | Toxicity | N/A | Bias | T | 4800 | 2023 | 8.3.2 [256] |
| SafetyBench | Comprehensive | Safety | N/A | T | 11435 | 2023 | 8.3.2 [254] |
| Cvalues (Chinese) | Comprehensive | Safety | N/A | T | 4312 | 2023 | 8.3.2 [230] |
| DecodingTrust | Comprehensive | Safety | N/A | F | N/A | 2023 | 8.3.2 [201] |
| KoLA | Comprehensive | Comprehension & Reasoning | N/A | F | N/A | 2023 | 8.3.2 [236] |
| BIG-bench | Comprehensive | Reasoning & Safety | N/A | F | N/A | 2022 | 8.3.2 [182] |

</details>

AutoAWQ 相关代码的原版权与使用条件见 [AutoAWQ-LICENSE](AutoAWQ-LICENSE)。调研表和研究链接已合并到本页，PDF 保留独立阅读版本。
