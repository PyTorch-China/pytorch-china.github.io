# 在 Linux 上安装
{:.no_toc}

PyTorch 可以在各种 Linux 发行版上安装和使用。根据您的系统和计算需求，您在 Linux 上使用 PyTorch 的体验可能在处理时间方面有所不同。建议（但不是必需）您的 Linux 系统有 NVIDIA 或 AMD GPU，以便充分利用 PyTorch 的 [CUDA](https://developer.nvidia.com/cuda-zone) [支持](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html?highlight=cuda#cuda-tensors) 或 [ROCm](https://docs.amd.com) 支持。


## 先决条件
{: #linux-prerequisites}

### 支持的 Linux 发行版

PyTorch 支持使用 [glibc](https://www.gnu.org/software/libc/) >= v2.17 的 Linux 发行版，包括以下：

* [Arch Linux](https://www.archlinux.org/download/)，最低版本 2012-07-15
* [CentOS](https://www.centos.org/download/)，最低版本 7.3-1611
* [Debian](https://www.debian.org/distrib/)，最低版本 8.0
* [Fedora](https://getfedora.org/)，最低版本 24
* [Mint](https://linuxmint.com/download.php)，最低版本 14
* [OpenSUSE](https://software.opensuse.org/)，最低版本 42.1
* [PCLinuxOS](https://www.pclinuxos.com/)，最低版本 2014.7
* [Slackware](http://www.slackware.com/getslack/)，最低版本 14.2
* [Ubuntu](https://www.ubuntu.com/download/desktop)，最低版本 13.04

> 这里的安装说明通常适用于所有支持的 Linux 发行版。一个例外是您的发行版可能支持 `yum` 而不是 `apt`。所示的具体示例是在 Ubuntu 18.04 机器上运行的。

### Python
{: #linux-python}

Python 3.8-3.11 通常默认安装在我们支持的任何 Linux 发行版上，这符合我们的建议。

> 提示：默认情况下，您必须使用命令 `python3` 来运行 Python。如果您想只使用命令 `python`，而不是 `python3`，您可以将 `python` 符号链接到 `python3` 二进制文件。

但是，如果您想安装另一个版本，有多种方法：

* APT
* [Python website](https://www.python.org/downloads/mac-osx/)

如果您决定使用 APT，可以运行以下命令来安装：

```bash
sudo apt install python
```

> 如果您使用 [Anaconda](#anaconda) 安装 PyTorch，它将安装一个沙盒版本的 Python，用于运行 PyTorch 应用程序。



### 包管理器
{: #linux-package-manager}

要安装 PyTorch 二进制文件，您需要使用两个支持的包管理器之一：[Anaconda](https://www.anaconda.com/download/#linux) 或 [pip](https://pypi.org/project/pip/) 。推荐使用 Anaconda 作为包管理器，因为它将在一个沙盒安装中提供所有 PyTorch 依赖项，包括 Python。

#### Anaconda

要安装 Anaconda，您将使用 [命令行安装程序](https://www.anaconda.com/download/#linux) 。右键单击 64 位安装程序链接，选择 `复制链接地址` ，然后使用以下命令：

```bash
# The version of Anaconda may be different depending on when you are installing`
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
# and follow the prompts. The defaults are generally good.`
```

> 您可能需要打开一个新的终端或重新加载 `~/.bashrc` 以获得对 `conda` 命令的访问权限。

#### pip

*Python 3*

虽然 Python 3.x 默认安装在 Linux 上，但 `pip` 默认不安装。

```bash
sudo apt install python3-pip
```

> 提示：如果您想只使用命令 `pip` ，而不是 `pip3` ，您可以将 `pip` 符号链接到 `pip3` 二进制文件。

## 安装
{: #linux-installation}

### Anaconda
{: #linux-anaconda}

#### 无 CUDA/ROCm

如果您通过 Anaconda 安装 PyTorch，并且没有 [CUDA-capable](https://developer.nvidia.com/cuda-zone) or [ROCm-capable](https://docs.amd.com) ，或不需要 CUDA/ROCm（即 GPU 支持），在上面的选择器中，选择操作系统：Linux，包：Conda，语言：Python 和计算平台：CPU。 然后，运行向您显示的命令。

#### 有 CUDA

如果您通过 Anaconda 安装 PyTorch，并且有 [CUDA-capable](https://developer.nvidia.com/cuda-zone) 系统，在上面的选择器中，选择操作系统：Linux，包：Conda 和适合您机器的 CUDA 版本。通常，最新的 CUDA 版本更好。 然后，运行向您显示的命令。

#### 有 ROCm

目前不支持通过 Anaconda 安装带 ROCm 的 PyTorch。请改用 pip。


### pip
{: #linux-pip}

#### No CUDA

如果您通过 pip 安装 PyTorch，并且没有 [CUDA-capable](https://developer.nvidia.com/cuda-zone) 或 [ROCm-capable](https://docs.amd.com)  系统，或不需要 CUDA/ROCm（即 GPU 支持），在上面的选择器中，选择操作系统：Linux，包：Pip，语言：Python 和计算平台：CPU。 然后，运行向您显示的命令。

#### With CUDA

如果您通过 pip 安装 PyTorch，并且有 [CUDA-capable](https://developer.nvidia.com/cuda-zone) 系统，在上面的选择器中，选择操作系统：Linux，包：Pip，语言：Python 和适合您机器的 CUDA 版本。通常，最新的 CUDA 版本更好。 然后，运行向您显示的命令。


#### With ROCm

如果您通过 pip 安装 PyTorch，并且有 [ROCm-capable](https://docs.amd.com)  系统，在上面的选择器中，选择操作系统：Linux，包：Pip，语言：Python 和支持的 ROCm 版本。 然后，运行向您显示的命令。



## Verification
{: #linux-verification}

为确保 PyTorch 安装正确，我们可以通过运行示例 PyTorch 代码来验证安装。这里我们将构造一个随机初始化的张量。

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

此外，要检查您的 GPU 驱动程序和 CUDA/ROCm 是否已启用并可被 PyTorch 访问，运行以下命令以返回 GPU 驱动程序是否已启用（PyTorch 的 ROCm 构建在 Python API 级别 [link](https://github.com/pytorch/pytorch/blob/master/docs/source/notes/hip.rst#hip-interfaces-reuse-the-cuda-interfaces) 使用相同的语义 链接，因此以下命令也应适用于 ROCm）：


```python
import torch
torch.cuda.is_available()
```

## 从源代码构建
{: #linux-from-source}

对于大多数 PyTorch 用户来说，通过包管理器从预构建的二进制文件安装将提供最佳体验。但是，有时您可能想安装最新的 PyTorch 代码，无论是为了测试还是实际开发 PyTorch 核心。要安装最新的 PyTorch 代码，您需要从 [源代码构建 PyTorch](https://github.com/pytorch/pytorch#from-source) 。



### 先决条件
{: #linux-prerequisites-2}

1. 安装 [Anaconda](#anaconda) 或 [Pip](#pip) 如果您需要构建带 GPU 支持的 PyTorch 

    a. 对于 NVIDIA GPU，如果您的机器有 CUDA 兼容 GPU，请安装 [CUDA](https://developer.nvidia.com/cuda-downloads) 。 
    b. 对于 AMD GPU，如果您的机器有 ROCm 兼容 GPU，请安装 [ROCm](https://docs.amd.com)。

按照此处描述的步骤操作：[https://github.com/pytorch/pytorch#from-source](https://github.com/pytorch/pytorch#from-source)