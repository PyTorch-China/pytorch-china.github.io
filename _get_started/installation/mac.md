# 在 macOS 上安装
{:.no_toc}

PyTorch 可以在 macOS 上安装和使用。根据您的系统和 GPU 功能，您在 Mac 上使用 PyTorch 的体验可能在处理时间方面有所不同。

## 先决条件
{: #mac-prerequisites}

### macOS 版本

PyTorch 支持 macOS 10.15（Catalina）或更高版本。

### Python
{: #mac-python}

建议您使用 Python 3.8 - 3.11。
您可以通过 Anaconda 包管理器（见[下文](#anaconda)）、[Homebrew](https://brew.sh/) 或
[Python 官网](https://www.python.org/downloads/mac-osx/)安装 Python。

In one of the upcoming PyTorch releases, support for Python 3.8 will be deprecated.

### 包管理器
{: #mac-package-manager}

要安装 PyTorch 二进制文件，您需要使用两种受支持的包管理器之一：[Anaconda](https://www.anaconda.com/download/#macos) 或 [pip](https://pypi.org/project/pip/)。推荐使用 Anaconda 作为包管理器，因为它将在一个沙盒安装中提供所有 PyTorch 依赖项，包括 Python。

#### Anaconda

要安装 Anaconda，您可以[下载图形安装程序](https://www.anaconda.com/download/#macos)或使用命令行安装程序。如果您使用命令行安装程序，可以右键单击安装程序链接，选择"复制链接地址"，或在 Intel Mac 上使用以下命令：

```bash
# The version of Anaconda may be different depending on when you are installing`
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
sh Miniconda3-latest-MacOSX-x86_64.sh
# and follow the prompts. The defaults are generally good.`
```

或在 M1 Mac 上使用以下命令：

```bash
# The version of Anaconda may be different depending on when you are installing`
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
sh Miniconda3-latest-MacOSX-arm64.sh
# and follow the prompts. The defaults are generally good.`
```
#### pip

*Python 3*

如果您通过 Homebrew 或 Python 官网安装了 Python，pip 已随之安装。如果您安装了 Python 3.x，那么您将使用 pip3 命令。

> 提示：如果您想只使用 pip 命令，而不是 pip3，您可以将 pip 符号链接到 pip3 二进制文件。

## Installation
{: #mac-installation}

### Anaconda
{: #mac-anaconda}

要通过 Anaconda 安装 PyTorch，请使用以下 conda 命令：

```bash
conda install pytorch torchvision -c pytorch
```

### pip
{: #mac-anaconda}

要通过 pip 安装 PyTorch，请根据您的 Python 版本使用以下两个命令之一：

```bash
# Python 3.x
pip3 install torch torchvision
```

## Verification
{: #mac-verification}

为确保 PyTorch 安装正确，我们可以通过运行 PyTorch 示例代码来验证安装。这里我们将构造一个随机初始化的张量。

```python
import torch
x = torch.rand(5, 3)
print(x)
```

输出应该类似于：

```
tensor([[0.3380, 0.3845, 0.3217],
        [0.8337, 0.9050, 0.2650],
        [0.2979, 0.7141, 0.9069],
        [0.1449, 0.1132, 0.1375],
        [0.4675, 0.3947, 0.1426]])
```

## Building from source
{: #mac-from-source}

对于大多数 PyTorch 用户来说，通过包管理器安装预构建的二进制文件将提供最佳体验。但是，有时您可能想安装最新的 PyTorch 代码，无论是为了测试还是实际开发 PyTorch 核心。要安装最新的 PyTorch 代码，您需要从[源代码构建PyTorch](https://github.com/pytorch/pytorch#from-source)。




### Prerequisites
{: #mac-prerequisites-2}

1. [可选] 安装 [Anaconda](#anaconda)
2. 按照此处描述的步骤操作: [https://github.com/pytorch/pytorch#from-source](https://github.com/pytorch/pytorch#from-source)

您可以按照[上述方法](#mac-verification)验证安装结果
