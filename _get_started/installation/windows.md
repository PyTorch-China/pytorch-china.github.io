# 在 Windows 上安装
{:.no_toc}

PyTorch 可以在各种 Windows 发行版上安装和使用。根据您的系统和计算需求，您在 Windows 上使用 PyTorch 的体验可能在处理时间方面有所不同。建议（但不是必需）您的 Windows 系统有一个 NVIDIA GPU，以便充分利用 PyTorch 的 [CUDA](https://developer.nvidia.com/cuda-zone) [支持](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html?highlight=cuda#cuda-tensors)。

## 先决条件
{: #windows-prerequisites}

### 支持的 Windows 发行版

PyTorch 支持以下 Windows 发行版：

* [Windows](https://www.microsoft.com/en-us/windows) 7 及更高版本；推荐 [Windows 10](https://www.microsoft.com/en-us/software-download/windows10ISO) 或更高版本。
* [Windows Server 2008](https://docs.microsoft.com/en-us/windows-server/windows-server) r2 及更高版本

> 这里的安装说明通常适用于所有支持的 Windows 发行版。具体示例将在 Windows 10 企业版机器上运行。

### Python
{: #windows-python}

目前，Windows 上的 PyTorch 仅支持 Python 3.8-3.11；不支持 Python 2.x。

由于 Windows 默认不安装 Python，有多种方法可以安装 Python：

* [Chocolatey](https://chocolatey.org/)
* [Python 官网](https://www.python.org/downloads/windows/)
* [Anaconda](#anaconda)

> 如果您使用 Anaconda 安装 PyTorch，它将安装一个沙盒版本的 Python，用于运行 PyTorch 应用程序。

> 如果您决定使用 Chocolatey，并且尚未安装 Chocolatey，请确保以管理员身份运行命令提示符。

对于基于 Chocolatey 的安装，在管理员命令提示符中运行以下命令：

```bash
choco install python
```

### 包管理器
{: #windows-package-manager}

要安装 PyTorch 二进制文件，您需要使用至少两个支持的包管理器中的一个：[Anaconda](https://www.anaconda.com/download/#windows) 或 [pip](https://pypi.org/project/pip/) 。推荐使用 Anaconda 作为包管理器，因为它将在一个沙盒安装中提供所有 PyTorch 依赖项，包括 Python 和 pip。

#### Anaconda

要安装 Anaconda，您将使用 [安装程序](https://www.anaconda.com/download/#windows) 安装 PyTorch 3.x。点击安装程序链接并选择 `运行` 。Anaconda 将下载，并向您显示安装程序提示。默认选项通常是合理的。

#### pip

如果您通过 [上面](#windows-python) 推荐的任何方式安装了 Python，[pip](https://pypi.org/project/pip/) 已经为您安装好了。

## 安装
{: #windows-installation}

### Anaconda
{: #windows-anaconda}

要使用 Anaconda 安装 PyTorch，您需要通过 `Start | Anaconda3 | Anaconda Prompt` 打开 Anaconda。

#### 无 CUDA

如果要通过 Anaconda 安装 PyTorch，并且没有 [CUDA-capable](https://developer.nvidia.com/cuda-zone) 系统或不需要 CUDA，在上面的选择器中，选择 OS: Windows，Package: Conda 和 CUDA: None。 然后，运行向您显示的命令。

#### 有 CUDA

如果要通过 Anaconda 安装 PyTorch，并且您确实有 [CUDA-capable](https://developer.nvidia.com/cuda-zone) 系统，在上面的选择器中，选择 OS: Windows，Package: Conda 和适合您机器的 CUDA 版本。通常，最新的 CUDA 版本更好。 然后，运行向您显示的命令。


### pip
{: #windows-pip}

#### 无 CUDA

如果要通过 pip 安装 PyTorch，并且没有 [CUDA-capable](https://developer.nvidia.com/cuda-zone) 系统或不需要 CUDA，在上面的选择器中，选择 OS: Windows，Package: Pip 和 CUDA: None。 然后，运行向您显示的命令。

#### 有 CUDA

如果要通过 pip 安装 PyTorch，并且您确实有 [CUDA-capable](https://developer.nvidia.com/cuda-zone)  系统，在上面的选择器中，选择 OS: Windows，Package: Pip 和适合您机器的 CUDA 版本。通常，最新的 CUDA 版本更好。 然后，运行向您显示的命令。


## 验证
{: #windows-verification}

为了确保 PyTorch 安装正确，我们可以通过运行示例 PyTorch 代码来验证安装。这里我们将构造一个随机初始化的张量。

在命令行中，输入：

```bash
python
```

然后输入以下代码：

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

此外，要检查您的 GPU 驱动程序和 CUDA 是否已启用并可被 PyTorch 访问，运行以下命令以返回 CUDA 驱动程序是否已启用：

```python
import torch
torch.cuda.is_available()
```

## 源代码构建
{: #windows-from-source}

对于大多数 PyTorch 用户来说，通过包管理器从预构建的二进制文件安装将提供最佳体验。但是，有时您可能想安装最新的 PyTorch 代码，无论是为了测试还是实际开发 PyTorch 核心。要安装最新的 PyTorch 代码，您需要从 [源代码构建 PyTorch](https://github.com/pytorch/pytorch#from-source) 。

### 先决条件
{: #windows-prerequisites-2}

1. 安装 [Anaconda](#anaconda)
2. 安装 [CUDA](https://developer.nvidia.com/cuda-downloads), 如果机器有 [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus).
3. 如果您想在 Windows 上构建，还需要带有 MSVC 工具集的 Visual Studio 和 NVTX。这些依赖项的确切要求可以在查看 [这里](https://github.com/pytorch/pytorch#from-source) 。
4. 按照这里描述的步骤操作：[https://github.com/pytorch/pytorch#from-source](https://github.com/pytorch/pytorch#from-source)


您可以按照 [上面](#windows-verification) 描述的方法验证安装。

