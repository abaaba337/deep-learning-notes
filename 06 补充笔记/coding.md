# 第 06 章 补充笔记 · 代码与源码阅读

[学习路径](../README.md) · [概念与问答](concepts.md)

先看 6.6 的 PyTorch 对象与张量，再沿 6.7 的调用链阅读 AutoAWQ。下面三个 Python 代码块可分别在 CPU 环境独立运行；量化项目的 GPU 配置与命令只在实践 README 维护。

## 6.6 Python 与 PyTorch 常用接口

### 6.6.1 模块遍历、参数与缓冲区

| 接口 | 用法与边界 |
|---|---|
| `children()` / `named_children()` | 迭代直接子模块，后者产生名称和模块对 |
| `modules()` / `named_modules()` | 递归遍历，包括模型自身；默认去掉重复模块 |
| `get_submodule("0.1")` | 按点分隔的路径访问子模块，不是接收整个 `(name, module)` 元组 |
| `get_parameter(name)` / `get_buffer(name)` | 按路径取得参数或缓冲区；`getattr(model, "a.b")` 不会递归访问 a.b |
| `state_dict()` | 包括参数（含冻结参数）和持久缓冲区；不是整个 Python 模型，返回张量通常共享底层存储 |
| `register_buffer(name, value)` | 注册随模型迁移设备的张量；`persistent=False` 时不写入 state_dict |

Buffer 不属于优化器通常读取的 `parameters()`，但注册操作本身不禁止 autograd。普通 Tensor 属性不会自动随模型移动或保存；`nn.Parameter` 属性会自动注册。调用 `model(x)` 才会走模块的调用逻辑和 hooks，通常不要直接调用 `model.forward(x)`。参见 [PyTorch Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)。

```python
import torch
from torch import nn

model = nn.Sequential(nn.Linear(3, 2), nn.ReLU())
model[0].weight.requires_grad_(False)
model.register_buffer("scale", torch.ones(2))
model.register_buffer("scratch", torch.zeros(2), persistent=False)
assert next(model.modules()) is model
assert model.get_submodule("0") is model[0]
assert model.get_parameter("0.weight") is model[0].weight
assert "0.weight" in model.state_dict()  # 冻结参数仍会保存
assert "scale" in model.state_dict() and "scratch" not in model.state_dict()
x = torch.randn(4, 3)
torch.testing.assert_close(model[0](x), x @ model[0].weight.T + model[0].bias)
```

### 6.6.2 检查签名与收集特征

`inspect.signature(callable).parameters` 是参数名到 Parameter 的有序映射；`.default is inspect.Parameter.empty` 表示没有默认值。部分内建函数不提供可检查签名。`defaultdict(list)` 在访问缺失键时才创建列表；`dir(instance)` 可辅助发现属性，但不是完整的接口文档。

```python
import inspect
from collections import defaultdict

def forward(x, scale=1):
    return x * scale

params = inspect.signature(forward).parameters
assert params["x"].default is inspect.Parameter.empty
assert params["scale"].default == 1
features = defaultdict(list)
features["layer"].append([1, 2])
assert list(features) == ["layer"]
```

### 6.6.3 张量操作与内存

`clamp(min, max)` 截断值域；`amax/amin(dim, keepdim=True)` 保留约简维度方便广播。`W.mul_(s.view(1, -1))` 对二维 W 的每列缩放，下划线表示原地修改；训练时要考虑叶子参数与反向传播保存的值，不能靠 `.data` 绕开 autograd 检查。

`gc.collect()` 主要收集不可达的循环引用对象，不保证系统占用立刻下降。`torch.cuda.empty_cache()` 释放分配器中未使用的缓存块，不释放活跃张量；不要默认在每个 batch 调用它。持有图、列表或 hook 引用的张量，需要先结束引用生命周期。参见 [CUDA empty_cache](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.empty_cache.html)。

<a id="awq"></a>

## 6.7 AutoAWQ 源码阅读

### 6.7.1 对象与调用链

以下名称对应 **AutoAWQ 0.2.7.post3**，属于历史实验接口，不是跨版本承诺。`awq_model` 是包装对象，`awq_model.model` 才是 Transformers/PyTorch 模型。LLaMA 的 `model.model.layers` 是 decoder blocks，不是 encoder blocks。

```text
AutoAWQForCausalLM.from_pretrained → 具体架构包装类
  → BaseAWQForCausalLM.from_pretrained（浮点模型、配置、可选 processor）
  → quantize → AwqQuantizer.__init__ → init_quant
      → get_calib_dataset → 捕获首个 decoder block 的输入和 kwargs
  → AwqQuantizer.quantize（逐 block）
      → _get_input_feat → _search_best_scale → apply_scale
      → _search_best_clip → apply_clip → _apply_quant
  → save_quantized（权重分片、配置；必要时另存 tokenizer/processor）
```

`export_compatible=True` 时先保留兼容导出的浮点形式，`pack()` 再遍历目标层执行 `_apply_quant`，转换为后端的低比特存储。`save_quantized()` 保存当时的模型状态，不等于自动执行所有量化步骤，也不应误认为直接产生任意格式的 GGUF 文件。原接口可能覆盖同名输出，本仓实践入口另行拒绝覆盖。

| 位置 | 阅读重点 |
|---|---|
| `BaseAWQForCausalLM` | `model_type/config/quant_config/is_quantized` 记录模型与量化状态；`processor` 是可选处理器，不一定只处理图像；`to/forward/generate` 委托底层模型 |
| `from_quantized` | 根据配置建立量化结构并载入权重，区别于浮点加载 |
| `_load_config`、`_load_quantized_modules`、`_scale_activations` | 找配置与路径、替换量化 Linear、恢复所需 ScaledActivation |
| `LlamaAWQForCausalLM` | `get_model_layers` 选 decoder blocks；`move_embed` 迁移嵌入；`get_act_for_scaling/get_layers_for_scaling` 描述相邻算子的缩放关系 |
| `fuse_layers` | 针对后端融合受支持算子以便推理；不是重新训练，也不是所有架构都支持 |
| `LlamaDecoderLayer` / `LlamaForCausalLM` | 前者是单个 block，后者是含语言建模输出头的完整模型 |

### 6.7.2 校准输入与配置

`get_calib_dataset` 接收数据集名称、字符串列表或 token ID 列表；`split/text_column` 指定数据切分和文本列。样本可以是 `{"text": "一段校准文字", "meta": {"pile_set_name": "Pile-CC"}}`。它筛选并 tokenize 文本，拼接后切成固定长度的 token 块，丢弃不足一块的尾部；每块不是一条原始文本，更不是已经计算好的 embedding。

| 配置或状态 | 含义 |
|---|---|
| `w_bit/group_size/zero_point/version` | 权重位数、分组大小、是否使用零点、量化内核/布局版本 |
| `calib_data/split/text_column` | 校准语料来源与字段 |
| `max_calib_samples=128` / `max_calib_seq_len=512` | 历史默认的筛选文本数量和 token 块长度；筛选文本数不等于最终块数 |
| `n_parallel_calib_samples` | 校准前向并行处理的样本数 |
| `max_chunk_memory=1024**3` | 某些统计和损失计算的分块预算，不是整个模型的显存上限 |
| `duo_scaling/apply_clip` | 是否结合权重统计搜索缩放、是否搜索裁剪阈值 |
| `modules_to_not_convert/export_compatible` | 跳过指定层、延后打包的导出模式 |
| `modules/module_kwargs/inps` | 待处理 blocks、当前输入关键字参数、当前隐藏状态 |

`init_quant` 临时用 Catcher 包装首层，取得 embedding 后的隐藏状态和 kwargs，然后恢复首层。`_get_input_feat` 注册前向 hooks，收集目标线性层输入并解除 hooks；通常具有“样本、token、通道”维度，实际形状依层而定。`_sanitize_kwargs` 依据 forward 签名筛选参数。读代码时检查 hooks 是否移除、临时层是否恢复、保存的特征是否仍占用 GPU。

### 6.7.3 缩放、伪量化与裁剪

采用 PyTorch 行向量约定，X 的末维是输入通道，W 的形状为 `(out_features, in_features)`。令各通道缩放 s > 0：

$$XW^\top=(X/s)(W\odot s)^\top.$$

量化后等式通常不再精确成立；搜索 s 的目标是减小输出重构均方误差。`_module_forward` 计算被检查模块输出，`_compute_loss` 分块累计平方差后除以元素总数；它也可以比较整个被检查子模块的输出，不仅限于一个 Linear。

```python
import torch

torch.manual_seed(0)
x, w = torch.randn(5, 4), torch.randn(3, 4)
s = torch.tensor([0.5, 1.0, 2.0, 4.0])
scaled_w = w * s.view(1, -1)
torch.testing.assert_close(x @ w.T, (x / s) @ scaled_w.T)
# 演示每行一组的非对称 4-bit 伪量化；没有真实整数打包。
lo, hi = scaled_w.amin(1, keepdim=True), scaled_w.amax(1, keepdim=True)
step = (hi - lo).clamp(min=1e-5) / 15
zero = (-torch.round(lo / step)).clamp(0, 15)
q = (torch.round(scaled_w / step) + zero).clamp(0, 15)
reconstructed_w = (q - zero) * step
assert q.min() >= 0 and q.max() <= 15
mse = ((x / s) @ reconstructed_w.T - x @ w.T).square().mean()
assert torch.isfinite(mse)
print("Quantization output MSE:", mse.item())
```

`_search_best_scale/_compute_best_scale` 搜索离散指数 r。激活统计来自每通道绝对值均值；duo_scaling 还结合权重统计，近似构造 `x_mean**r / (w_mean**(1-r) + eps)`，再钳制并除以 `sqrt(s.max()*s.min())`。这一步控制尺度，不是均值为 0、方差为 1 的标准化。候选权重伪量化、计算输出误差后必须恢复原权重，不能让候选之间互相污染。

`pseudo_quantize_tensor` 将分组权重映射到有限整数等级再反量化，返回浮点重构值、scale 和 zero point；仍占浮点存储。`pseudo_dequantize_tensor` 则是由量化数据和元信息恢复近似浮点值的辅助过程。

`_search_best_clip/_compute_best_clip` 在若干收缩阈值中比较重构误差，返回每层阈值；分组乘积通常沿 group_size 求和。这里比较的是已匹配缩放的输入与权重，不是对训练梯度做裁剪。FP32 是该处用于累计误差的精度，不是计算机能表示的“最高精度”。

### 6.7.4 应用缩放、打包与辅助函数

`awq.quantize.scale.apply_scale` 按相邻算子结构分派到 `scale_ln_fcs`、`scale_fc_fc`、`scale_fc_fcs` 或 `scale_gelu_fc`，并同步更新供 clipping 使用的输入特征。`ScaledActivation` 在激活后补偿除以 s，不能假设 GELU 自身满足任意缩放等价。`apply_clip` 才将选中的阈值应用到权重。

`_apply_quant` 按后端用 `WQLinear.from_linear` 替换原 Linear。`qweight/qzeros/scales` 保存打包权重、零点与缩放信息，不是把零点也叫 qweight。4-bit 打包可将八个 4-bit 字段放进一个 32-bit 整数，涉及 `<<` 位移和 `|` 按位或；各后端排序、分组和布局有差异，不能把教学示例当作通用内核格式。`from_linear` 是 classmethod，可通过类调用工厂式构造，不需要先创建量化层实例。

| 模块 | 辅助函数及用途 |
|---|---|
| `awq.utils.module` | `try_import` 可选导入；`get_named_linears` 枚举 Linear；`get_op_by_name/set_op_by_name/get_op_name` 查找替换；`append_str_prefix` 补名称前缀；`exclude_layers_to_not_quantize` 过滤层 |
| `awq.utils.utils` | `get_module_by_name_suffix/set_module_name` 定位模块；`simple_dispatch_model` 分配模型；`clear_memory` 清理缓存；`compute_memory_used_pct/get_best_device/get_lowest_memory_device_index` 查询设备与内存 |

源码依据：[AutoAWQ 0.2.7.post3](https://pypi.org/project/autoawq/0.2.7.post3/)。本仓的实际入口、语言层校准和 KV cache 修复见 [实践 README](../03%20Transformer与大模型入门/practices/mllm-compression-safety/README.md) 与其中的 `patch_autoawq.py`，不再保留整份安装包差异作为第二套实现。
