# PyTorch 升级到 CUDA 12.8/13.0 指南

## 当前环境检查

以下是历史机器的 `nvidia-smi` 记录，不代表当前环境；该工具显示的是驱动支持的 CUDA 版本上限，不是已安装的 PyTorch CUDA 运行时版本：
- **NVIDIA 驱动版本**: 573.24
- **支持的 CUDA 版本**: 12.8
- **GPU**: NVIDIA GeForce RTX 5070

## 升级步骤

### 1. 检查当前 PyTorch 版本

运行以下 Python 代码检查当前版本：

```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
```

### 2. 卸载旧版本（可选但推荐）

```bash
pip uninstall torch torchvision torchaudio
```

### 3. 安装支持 CUDA 12.8 的 PyTorch

**方法一：使用 pip（推荐）**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**方法二：使用 conda 环境并通过 pip 安装**

```bash
conda create -n dl-notes python=3.11
conda activate dl-notes
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. 关于 CUDA 13.0

CUDA 13.0 构建已出现在官方发布中；是否适合你的平台、Python、GPU 与驱动，应以 [官方安装选择器](https://pytorch.org/get-started/locally/) 为准。不要仅根据版本号更大就升级，也不要假定驱动 573.24 可以运行所有 CUDA 13.0 构建。

### 5. 验证安装

安装完成后，运行以下代码验证：

```python
import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"GPU 设备名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    
    # 测试 GPU 计算
    x = torch.randn(3, 3).cuda()
    print(f"GPU 张量测试: {x.device}")
    print("✓ PyTorch CUDA 安装成功！")
else:
    print("✗ CUDA 不可用，请检查安装")
```

### 6. 常见问题

**Q: 安装后 CUDA 仍然不可用？**
- 核对当前 NVIDIA 驱动、GPU 架构与所选 PyTorch 构建的兼容要求，不能仅凭历史驱动版本判断
- 重启 Python 环境或 Jupyter 内核
- 检查是否有多个 PyTorch 版本冲突

**Q: 如何指定特定版本的 PyTorch？**
下面只是历史版本配套示例，不适用于需要更新架构支持的 RTX 50 系列；实际选择请核对 [官方版本表](https://docs.pytorch.org/get-started/previous-versions/)。
```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

**Q: 需要单独安装 CUDA 工具包吗？**
- 运行官方二进制包通常不需要另装 CUDA Toolkit；编译 PyTorch 或自定义 CUDA 扩展时仍可能需要
- 只需确保 NVIDIA 驱动支持即可

## 快速升级命令（一键执行）

```bash
# 卸载旧版本
pip uninstall torch torchvision torchaudio -y

# 安装 CUDA 12.8 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

