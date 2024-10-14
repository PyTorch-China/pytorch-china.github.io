---
layout: get_started
title: PyTorch 2.0
permalink: /get-started/pytorch-2.0/
featured-img: "assets/images/featured-img-pytorch-2.png"
background-class: get-started-background
body-class: get-started
order: 2
published: true
---

## 概述

介绍 PyTorch 2.0，这是我们迈向 PyTorch 下一代 2 系列版本的第一步。在过去的几年里，我们从 PyTorch 1.0 创新和迭代到最近的 1.13，并转移到新成立的 PyTorch 基金会，成为 Linux 基金会的一部分。

除了我们惊人的社区之外，PyTorch 的最大优势在于我们继续作为一流的 Python 集成，命令式风格，API 的简单性和选项。PyTorch 2.0 提供了相同的 eager-mode 开发和用户体验，同时从根本上改变并超充 PyTorch 在编译器级别下的操作方式。我们能够提供更快的性能和对动态形状和分布式的支持。

在下面，您将找到所有需要的信息，以更好地了解 PyTorch 2.0 是什么，它的发展方向，更重要的是如何今天就开始使用（例如，教程，要求，模型，常见问题）。还有很多需要学习和开发的内容，但我们期待社区的反馈和贡献，以使 2 系列更好，并感谢所有使 1 系列如此成功的人。

## PyTorch 2.x：更快，更 Python 化，并且一如既往地动态

今天，我们宣布 `torch.compile`，这是一个将 PyTorch 性能推向新高度的功能，并开始将 PyTorch 的部分内容从 C++ 移回 Python。我们认为这是 PyTorch 的一个重大新方向——因此我们称之为 2.0。`torch.compile` 是一个完全附加（和可选）的功能，因此 2.0 从定义上是 100% 向后兼容的。

支撑 `torch.compile` 的是新技术——TorchDynamo、AOTAutograd、PrimTorch 和 TorchInductor。

- **TorchDynamo** 使用 Python Frame Evaluation Hooks 安全地捕获 PyTorch 程序，这是我们 5 年 R&D 研究安全图捕获的重大创新成果。

- **AOTAutograd** 通过生成提前的反向跟踪来重载 PyTorch 的 autograd 引擎，作为跟踪自动微分。

- **PrimTorch** 将 ~2000+ PyTorch 操作符规范化为一个封闭集的 ~250 个原始操作符，开发人员可以针对这些操作符构建完整的 PyTorch 后端。这大大降低了编写 PyTorch 功能或后端的门槛。

- **TorchInductor** 是一个深度学习编译器，可以为多个加速器和后端生成快速代码。对于 NVIDIA 和 AMD GPU，它使用 OpenAI Triton 作为关键构建块。

TorchDynamo、AOTAutograd、PrimTorch 和 TorchInductor 是用 Python 编写的，并支持动态形状（即能够发送不同大小的张量而不引起重新编译），使它们灵活、易于修改并降低了开发人员和供应商的进入门槛。

为了验证这些技术，我们使用了各种机器学习领域的 163 个开源模型。我们仔细构建了这个基准测试，包括图像分类、目标检测、图像生成、各种 NLP 任务（如语言建模、问答、序列分类）、推荐系统和强化学习等任务。我们将基准测试分为三类：

<ul style="margin: 1.5rem 0 1.5rem 0;">
  <li>46 个来自 <a href="https://github.com/huggingface/transformers" target="_blank">HuggingFace Transformers</a> 的模型</li>
  <li>61 个来自 <a href="https://github.com/rwightman/pytorch-image-models" target="_blank">TIMM</a> 的模型：由 Ross Wightman 提供的最先进的 PyTorch 图像模型集合</li>
  <li>56 个来自 <a href="https://github.com/pytorch/benchmark/" target="_blank">TorchBench</a> 的模型：从 GitHub 上精选的流行代码库</li>
</ul>

[...]

<p>
<img src="/assets/images/pytorch-2.0-img2.png" width="90%">
</p>

### 用户评价

以下是一些 PyTorch 用户对我们新方向的评价：

**Sylvain Gugger**，**HuggingFace transformers 的主要维护者**：

_"只需添加一行代码，PyTorch 2.0 就能在训练 Transformers 模型时提供 1.5 倍到 2 倍的加速。这是自混合精度训练引入以来最令人兴奋的事情！"_

**Ross Wightman**，**TIMM 的主要维护者**（PyTorch 生态系统中最大的视觉模型中心之一）：

_“它在大多数 TIMM 模型的推理和训练工作负载中开箱即用，无需代码更改”_

**Luca Antiga**，**Lightning AI 的 CTO** 以及 **PyTorch Lightning 的主要维护者之一**：

_“PyTorch 2.0 体现了深度学习框架的未来。能够在几乎不需要用户干预的情况下捕获 PyTorch 程序，并在设备上获得大幅加速和程序操作，开箱即用，为 AI 开发人员解锁了一个全新的维度。”_

## 动机

我们对 PyTorch 的理念一直是将灵活性和可修改性作为我们的首要任务，性能紧随其后。我们努力实现：

1. 高性能的 eager 执行
2. Python 化的内部结构
3. 良好的分布式、自动微分、数据加载、加速器等抽象

自 2017 年推出 PyTorch 以来，硬件加速器（如 GPU）的计算速度提高了约 15 倍，内存访问速度提高了约 2 倍。因此，为了保持 eager 执行的高性能，我们不得不将 PyTorch 内部的相当一部分移到 C++。将内部结构移到 C++ 使其不易修改，并增加了代码贡献的门槛。

从第一天起，我们就知道 eager 执行的性能极限。2017 年 7 月，我们开始了第一个研究项目，开发 PyTorch 的编译器。编译器需要使 PyTorch 程序变得更快，但不能以牺牲 PyTorch 体验为代价。我们的关键标准是保持某些灵活性——支持研究人员在各种探索阶段使用的动态形状和动态程序。

<p>
<img src="/assets/images/pytorch-2.0-img3.gif" width="90%">
</p>

## 技术概述

多年来，我们在 PyTorch 内部构建了几个编译器项目。让我们将编译器分为三部分：

- 图获取
- 图降低
- 图编译

图获取是构建 PyTorch 编译器时的更大挑战。

在过去的 5 年里，我们构建了 `torch.jit.trace`、TorchScript、FX 跟踪、Lazy Tensors。但没有一个能满足我们的所有需求。有些灵活但不快，有些快但不灵活，有些既不快也不灵活。有些用户体验很差（例如，默默地出错）。虽然 TorchScript 很有前途，但它需要对您的代码及其依赖的代码进行大量更改。这种对代码进行大量更改的需求使得它对许多 PyTorch 用户来说是不可行的。

<p>
<img src="/assets/images/pytorch-2.0-img4.jpg" width="90%">
<center><u>PyTorch 编译过程</u></center>
</p>

### TorchDynamo：可靠且快速地获取图

今年早些时候，我们开始研究 TorchDynamo，这是一种使用 [PEP-0523](https://peps.python.org/pep-0523/) 中引入的 CPython 功能（称为 Frame Evaluation API）的方法。我们采用数据驱动的方法来验证其在图捕获方面的有效性。我们使用了 7000 多个用 PyTorch 编写的 GitHub 项目作为验证集。虽然 TorchScript 和其他方法在 50% 的时间里甚至难以获取图，通常伴随着很大的开销，但 TorchDynamo 在 [99% 的时间里](https://dev-discuss.pytorch.org/t/torchdynamo-update-8-torchdynamo-passed-correctness-check-on-7k-github-models/663) 正确、安全且几乎没有开销地获取了图——无需对原始代码进行任何更改。这时我们知道，我们终于突破了多年来在灵活性和速度方面的障碍。

### TorchInductor：使用 define-by-run IR 的快速代码生成

对于 PyTorch 2.0 的新编译器后端，我们从用户编写高性能自定义内核的方式中汲取了灵感：越来越多地使用 [Triton](https://github.com/openai/triton) 语言。我们还希望编译器后端使用与 PyTorch eager 类似的抽象，并且足够通用以支持 PyTorch 的广泛功能。TorchInductor 使用 Python 化的 define-by-run 循环级 IR 将 PyTorch 模型自动映射到 GPU 上生成的 Triton 代码和 CPU 上的 C++/OpenMP 代码。TorchInductor 的核心循环级 IR 仅包含 ~50 个操作符，并且用 Python 实现，使其易于修改和扩展。

### AOTAutograd：重用 Autograd 进行提前图

对于 PyTorch 2.0，我们知道我们想要加速训练。因此，关键不仅是捕获用户级代码，还要捕获反向传播。此外，我们知道我们想要重用现有的经过实战测试的 PyTorch autograd 系统。AOTAutograd 利用 PyTorch 的 **torch_dispatch** 可扩展机制，通过我们的 Autograd 引擎进行跟踪，允许我们提前捕获反向传递。这使我们能够使用 TorchInductor 加速我们的前向和反向传递。

### PrimTorch：稳定的原始操作符

编写 PyTorch 后端是具有挑战性的。PyTorch 有 1200 多个操作符，如果考虑到每个操作符的各种重载，则有 2000 多个。

<p>
<img src="/assets/images/pytorch-2.0-img5.png" width="90%">
<center> <i> <u> 2000 多个 PyTorch 操作符的细分 </u></i> </center>
</p>

因此，编写后端或跨领域功能变成了一项耗费精力的工作。在 PrimTorch 项目中，我们正在定义更小且稳定的操作符集。PyTorch 程序可以一致地降低到这些操作符集。我们旨在定义两个操作符集：

- 约 ~250 个操作符的 Prim 操作符，它们相当低级。它们适合编译器，因为它们足够低级，需要将它们重新融合在一起以获得良好的性能。
- 约 ~750 个规范操作符的 ATen 操作符，适合按原样导出。它们适合已经在 ATen 级别集成的后端，或不需要编译以从较低级操作符集中恢复性能的后端。

我们在下面的开发人员/供应商体验部分中讨论了更多关于这个主题的内容。

## 用户体验

我们引入了一个简单的函数 `torch.compile`，它包装您的模型并返回一个编译的模型。

```python
compiled_model = torch.compile(model)
```

这个 `compiled_model` 持有对你的模型的引用，并将 `forward` 函数编译成更优化的版本。在编译模型时，我们提供了一些调整选项

```python
def torch.compile(model: Callable,
  *,
  mode: Optional[str] = "default",
  dynamic: bool = False,
  fullgraph:bool = False,
  backend: Union[str, Callable] = "inductor",
  # advanced backend options go here as kwargs
  **kwargs
) -> torch._dynamo.NNOptimizedModule
```

- **mode** 指定编译器在编译时应该优化的内容。

  - 默认模式是一个预设，尝试在不花费太多时间编译或使用额外内存的情况下高效编译。
  - 其他模式如  `reduce-overhead` 大大减少框架开销，但会消耗少量额外内存。`max-autotune` 编译时间较长，尝试生成最快的代码。

- **dynamic** 指定是否启用动态形状的代码路径。某些编译器优化不能应用于动态形状的程序。明确是否需要动态形状或静态形状的编译程序将帮助编译器生成更优化的代码。
- **fullgraph** 类似于 Numba 的 `nopython`。它将整个程序编译成一个图，或者抛出异常。大多数用户不需要使用此模式。如果你对性能非常敏感，可以尝试使用它。
- **backend** 指定使用哪个编译器后端。默认情况下使用 `TorchInductor`，但还有其他一些可选的后端。

<p>
<img src="/assets/images/pytorch-2.0-img6.png" width="90%">
</p>

编译体验旨在默认模式下提供最大收益和最大灵活性。以下是每种模式下的模型。

现在，让我们看一个编译模型并运行它的完整示例（使用随机数据）

```python
import torch
import torchvision.models as models

model = models.resnet18().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
compiled_model = torch.compile(model)

x = torch.randn(16, 3, 224, 224).cuda()
optimizer.zero_grad()
out = compiled_model(x)
out.sum().backward()
optimizer.step()
```

第一次运行 `compiled_model(x)` 时，它会编译模型。因此，运行时间较长。后续运行速度很快。

### Modes

编译器有一些预设，可以以不同方式调整编译模型。 
你可能正在运行一个由于框架开销而变慢的小模型。或者，你可能正在运行一个几乎无法放入内存的大模型。根据你的需求，你可能需要使用不同的模式。

```python
# API 尚未最终确定
# default: 优化大模型，低编译时间
#          并且不使用额外内存
torch.compile(model)

# reduce-overhead: 优化以减少框架开销
#                    并使用一些额外内存。帮助加速小模型
torch.compile(model, mode="reduce-overhead")

# max-autotune: 优化以生成最快的模型，
#               但编译时间非常长
torch.compile(model, mode="max-autotune")

```

### 读取和更新属性

访问模型属性的方式与 eager 模式下相同。 
你可以像通常一样访问或修改模型的属性（例如 model.conv1.weight），可以确保获取到正确的结果。TorchDynamo 在代码中插入 guard 以检查其假设是否成立。如果属性被修改，TorchDynamo 会自动重新编译。

```python
# optimized_model 的工作方式类似于 model，可以随意访问其属性并修改它们
optimized_model.conv1.weight.fill_(0.01)

# 这个更改会反映在 model 中
```

### Hooks

Module and Tensor [hooks](https://pytorch.org/docs/stable/notes/modules.html#module-hooks) don’t fully work at the moment, but they will eventually work as we finish development.
Module 和 Tensor [hooks](https://pytorch.org/docs/stable/notes/modules.html#module-hooks)  目前还为完全支持。

### 序列化

你可以序列化 `optimized_model` 或 `model` 的状态字典。它们指向相同的参数和状态，因此是等效的。

```python
torch.save(optimized_model.state_dict(), "foo.pt")
# 这两行代码结果相同
torch.save(model.state_dict(), "foo.pt")
```

你目前无法序列化 `optimized_model`。如果你希望直接保存对象，请保存 `model`。

```python
torch.save(optimized_model, "foo.pt") # Error
torch.save(model, "foo.pt")           # Works
```

### 推理和导出

对于模型推理，在使用 torch.compile 生成编译模型后，在实际模型服务之前运行一些预热步骤。这有助于减轻初始服务期间的延迟峰值。

此外，我们将引入一种称为 `torch.export` 的模式，该模式会仔细导出整个模型和 guard 基础设施，以适应需要保证和可预测延迟的环境。`torch.export` 需要对你的程序进行更改，特别是如果你有数据依赖的控制流。

```python
# API 尚未最终确定
exported_model = torch._dynamo.export(model, input)
torch.save(exported_model, "foo.pt")
```

这处于开发的早期阶段。有关更多详细信息，请观看 PyTorch 会议上的 Export Path 讲座。你也可以在本月开始的  “Ask the Engineers: 2.0 Live Q&A Series” 中参与此话题（更多详细信息在本文末尾）。

### Debugging Issues

A compiled mode is opaque and hard to debug. You will have questions such as:
编译模式隐晦且难以调试。你可能会有以下问题：

- 为什么我的程序在编译模式下崩溃?
- 编译模式的准确性是否与 eager 模式一样?
- 为什么我没有看到加速效果?

如果编译模式产生错误或崩溃，或与急切模式的结果不同（超出机器精度限制），那么很可能不是你的代码的问题。然而，了解哪个代码片段是问题的原因是有用的。

为了帮助调试和可重复性，我们创建了几个工具和日志功能，其中一个很有用：Minifier。

Minifier 会自动将你看到的问题缩小到一个小代码片段。这个小代码片段重现了原始问题，你可以用缩小后的代码提交一个 GitHub issue。这将帮助 PyTorch 团队轻松快速地修复问题。

如果你没有看到预期的加速效果，我们有 **torch._dynamo.explain** 工具，它可以解释你的代码的哪些部分导致了我们称之为“图中断”的情况。图中断通常会阻碍编译器加速代码，减少图中断的数量可能会加速你的代码（直到某个极限的收益递减）。

你可以在我们的 [故障排除指南](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html) 中找到更多信息。

### Dynamic Shapes

在支持 PyTorch 代码的通用性方面，一个关键要求是支持动态形状，并允许模型接受不同大小的张量而不需要每次形状变化时重新编译。

截至今天，对动态形状的支持是有限的，并且正在快速进展中。它将在稳定版本中完全实现。它被一个 `dynamic=True` 参数所控制，我们在一个功能分支（symbolic-shapes）上取得了更多进展，在该分支上我们已经成功地使用 TorchInductor 运行了 BERT_pytorch 训练，具有完全符号形状。对于具有动态形状的推理，我们有更多的覆盖。例如，让我们看看动态形状在语言模型文本生成中的常见设置。

我们可以看到，即使形状从 4 动态变化到 256，编译模式也能够始终如一地比 eager 模式快 40%。如果不支持动态形状，一个常见的解决方法是填充到最近的 2 的幂。然而，从下图可以看出，这会带来显著的性能开销，并且还会导致显著更长的编译时间。此外，填充有时很难正确完成。

By supporting dynamic shapes in PyTorch 2.0’s Compiled mode, we can get the best of performance _and_ ease of use.
通过在 PyTorch 2.0 的编译模式中支持动态形状，我们可以获得性能和易用性的最佳结合。

<div style="display:flex; flex-direction: row; padding: 10px;">
<img src="/assets/images/pytorch-2.0-img7.png" width="50%">
<img src="/assets/images/pytorch-2.0-img8.png" width="50%">
</div>

当前的工作进展非常迅速，我们可能会暂时让一些模型回归，因为我们正在进行基础设施的重大改进。关于动态形状的最新进展可以在 [这里](https://dev-discuss.pytorch.org/t/state-of-symbolic-shapes-branch/777/19)找到。

## Distributed

In summary, torch.distributed’s two main distributed wrappers work well in compiled mode.

Both `DistributedDataParallel` (DDP) and `FullyShardedDataParallel` (FSDP) work in compiled mode and provide improved performance and memory utilization relative to eager mode, with some caveats and limitations.

<p>
<center> <u>Speedups in AMP Precision</u></center>
<img src="/assets/images/pytorch-2.0-img9.png" width="90%">
<center><u>Left: speedups for FSDP in Compiled mode over eager mode (AMP precision).<br>
Right: FSDP in Compiled mode takes substantially lesser memory than in eager mode</u></center>
</p>

<div style="display:flex; flex-direction: row; padding:10px;">
<img src="/assets/images/pytorch-2.0-img10.png" width="50%">
<img src="/assets/images/pytorch-2.0-img11.png" width="50%">
</div>

### DistributedDataParallel (DDP)

DDP relies on overlapping AllReduce communications with backwards computation, and grouping smaller per-layer AllReduce operations into ‘buckets’ for greater efficiency. AOTAutograd functions compiled by TorchDynamo prevent communication overlap, when combined naively with DDP, but performance is recovered by compiling separate subgraphs for each ‘bucket’ and allowing communication ops to happen outside and in-between the subgraphs. DDP support in compiled mode also currently requires `static_graph=False`. See [this post](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860) for more details on the approach and results for DDP + TorchDynamo.

### FullyShardedDataParallel (FSDP)

FSDP itself is a “beta” PyTorch feature and has a higher level of system complexity than DDP due to the ability to tune which submodules are wrapped and because there are generally more configuration options. FSDP works with TorchDynamo and TorchInductor for a variety of popular models, if configured with the `use_original_params=True` flag. Some compatibility issues with particular models or configurations are expected at this time, but will be actively improved, and particular models can be prioritized if github issues are filed.

Users specify an `auto_wrap_policy` argument to indicate which submodules of their model to wrap together in an FSDP instance used for state sharding, or manually wrap submodules in FSDP instances. For example, many transformer models work well when each ‘transformer block’ is wrapped in a separate FSDP instance and thus only the full state of one transformer block needs to be materialized at one time. Dynamo will insert graph breaks at the boundary of each FSDP instance, to allow communication ops in forward (and backward) to happen outside the graphs and in parallel to computation.

If FSDP is used without wrapping submodules in separate instances, it falls back to operating similarly to DDP, but without bucketing. Hence all gradients are reduced in one operation, and there can be no compute/communication overlap even in Eager. This configuration has only been tested with TorchDynamo for functionality but not for performance.

## Developer/Vendor Experience

With PyTorch 2.0, we want to simplify the backend (compiler) integration experience. To do this, we have focused on **reducing the number of operators** and **simplifying the semantics** of the operator set necessary to bring up a PyTorch backend.

In graphical form, the PT2 stack looks like:

<p>
<img src="/assets/images/pytorch-2.0-img12.png" width="90%">
</p>

Starting in the middle of the diagram, AOTAutograd dynamically captures autograd logic in an ahead-of-time fashion, producing a graph of forward and backwards operators in FX graph format.

We provide a set of hardened decompositions (i.e. operator implementations written in terms of other operators) that can be leveraged to **reduce** the number of operators a backend is required to implement. We also **simplify** the semantics of PyTorch operators by selectively rewriting complicated PyTorch logic including mutations and views via a process called _functionalization_, as well as guaranteeing operator metadata information such as shape propagation formulas. This work is actively in progress; our goal is to provide a _primitive_ and _stable_ set of ~250 operators with simplified semantics, called _PrimTorch,_ that vendors can leverage (i.e. opt-in to) in order to simplify their integrations.  
After reducing and simplifying the operator set, backends may choose to integrate at the Dynamo (i.e. the middle layer, immediately after AOTAutograd) or Inductor (the lower layer).  We describe some considerations in making this choice below, as well as future work around mixtures of backends.

**Dynamo Backend**

Vendors with existing compiler stacks may find it easiest to integrate as a TorchDynamo backend, receiving an FX Graph in terms of ATen/Prims IR. Note that for both training and inference, the integration point would be immediately after AOTAutograd, since we currently apply decompositions as part of AOTAutograd, and merely skip the backward-specific steps if targeting inference.

**Inductor backend**

Vendors can also integrate their backend directly into Inductor. Inductor takes in a graph produced by AOTAutograd that consists of ATen/Prim operations, and further lowers them down to a loop level IR. Today, Inductor provides lowerings to its loop-level IR for pointwise, reduction, scatter/gather and window operations. In addition, Inductor creates fusion groups, does indexing simplification, dimension collapsing, and tunes loop iteration order in order to support efficient code generation. Vendors can then integrate by providing the mapping from the loop level IR to hardware-specific code. Currently, Inductor has two backends: (1) C++ that generates multithreaded CPU code, (2) Triton that generates performant GPU code. These Inductor backends can be used as an inspiration for the alternate backends.

**Mixture of Backends Interface (coming soon)**

We have built utilities for partitioning an FX graph into subgraphs that contain operators supported by a backend and executing the remainder eagerly. These utilities can be extended to support a “mixture of backends,” configuring which portions of the graphs to run for which backend. However, there is not yet a stable interface or contract for backends to expose their operator support, preferences for patterns of operators, etc. This remains as ongoing work, and we welcome feedback from early adopters.

## Final Thoughts

We are super excited about the direction that we’ve taken for PyTorch 2.0 and beyond. The road to the final 2.0 release is going to be rough, but come join us on this journey early-on. If you are interested in deep-diving further or contributing to the compiler, please continue reading below which includes more information on how to get started (e.g., tutorials, benchmarks, models, FAQs) and **Ask the Engineers: 2.0 Live Q&A Series** starting this month. Additional resources include:

- [Getting Started](https://pytorch.org/docs/stable/torch.compiler_get_started.html)
- [Tutorials](https://pytorch.org/tutorials/)
- [Documentation](https://pytorch.org/docs/stable)
- [Developer Discussions](https://dev-discuss.pytorch.org)

<script page-id="pytorch" src="{{ site.baseurl }}/assets/menu-tab-selection.js"></script>
<script src="{{ site.baseurl }}/assets/quick-start-module.js"></script>
<script src="{{ site.baseurl }}/assets/show-screencast.js"></script>

## Accelerating Hugging Face and TIMM models with PyTorch 2.0

Author: Mark Saroufim

`torch.compile()` makes it easy to experiment with different compiler backends to make PyTorch code faster with a single line decorator `torch.compile()`. It works either directly over an nn.Module as a drop-in replacement for torch.jit.script() but without requiring you to make any source code changes. We expect this one line code change to provide you with between 30%-2x training time speedups on the vast majority of models that you’re already running.

```python
opt_module = torch.compile(module)
```

torch.compile supports arbitrary PyTorch code, control flow, mutation and comes with experimental support for dynamic shapes. We’re so excited about this development that we call it PyTorch 2.0.

What makes this announcement different for us is we’ve already benchmarked some of the most popular open source PyTorch models and gotten substantial speedups ranging from 30% to 2x [https://github.com/pytorch/torchdynamo/issues/681](https://github.com/pytorch/torchdynamo/issues/681).

There are no tricks here, we’ve pip installed popular libraries like [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers), [https://github.com/huggingface/accelerate](https://github.com/huggingface/accelerate) and [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and then ran torch.compile() on them and that’s it.

It’s rare to get both performance and convenience, but this is why the core team finds PyTorch 2.0 so exciting.

## Requirements

For GPU (newer generation GPUs will see drastically better performance)

```
pip3 install numpy --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
```

For CPU

```
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

Optional: Verify Installation

```
git clone https://github.com/pytorch/pytorch
cd tools/dynamo
python verify_dynamo.py
```

Optional: Docker installation

We also provide all the required dependencies in the PyTorch nightly
binaries which you can download with

```
docker pull ghcr.io/pytorch/pytorch-nightly
```

And for ad hoc experiments just make sure that your container has access to all your GPUs

```
docker run --gpus all -it ghcr.io/pytorch/pytorch-nightly:latest /bin/bash
```

## Getting Started

Please read Mark Saroufim’s [full blog post](/blog/Accelerating-Hugging-Face-and-TIMM-models/) where he walks you through a tutorial and real models for you to try PyTorch 2.0 today.

Our goal with PyTorch was to build a breadth-first compiler that would speed up the vast majority of actual models people run in open source. The Hugging Face Hub ended up being an extremely valuable benchmarking tool for us, ensuring that any optimization we work on actually helps accelerate models people want to run.

The blog tutorial will show you exactly how to replicate those speedups so you can be as excited as to PyTorch 2.0 as we are. So please try out PyTorch 2.0, enjoy the free perf and if you’re not seeing it then please open an issue and we will make sure your model is supported [https://github.com/pytorch/torchdynamo/issues](https://github.com/pytorch/torchdynamo/issues)

After all, we can’t claim we’re created a breadth-first unless **YOUR** models actually run faster.

## FAQs  

1. **What is PT 2.0?**  
2.0 is the latest PyTorch version. PyTorch 2.0 offers the same eager-mode development experience, while adding a compiled mode via torch.compile. This compiled mode has the potential to speedup your models during training and inference.

2. **Why 2.0 instead of 1.14?**  
PyTorch 2.0 is what 1.14 would have been. We were releasing substantial new features that we believe change how you meaningfully use PyTorch, so we are calling it 2.0 instead.

3. **How do I install 2.0? Any additional requirements?**

    Install the latest nightlies:

    CUDA 11.8<br>

    ```
    pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118
    ```  

    CUDA 11.7  

    ```
    pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
    ```  

    CPU  

    ```
    pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cpu
    ```  

4. **Is 2.0 code backwards-compatible with 1.X?**  
Yes, using 2.0 will not require you to modify your PyTorch workflows. A single line of code `model = torch.compile(model)` can optimize your model to use the 2.0 stack, and smoothly run with the rest of your PyTorch code. This is completely opt-in, and you are not required to use the new compiler.

5. **Is 2.0 enabled by default?**  
2.0 is the name of the release. torch.compile is the feature released in 2.0, and you need to explicitly use torch.compile.

6. **How do I migrate my PT1.X code to PT2.0?**  
Your code should be working as-is without the need for any migrations. If you want to use the new Compiled mode feature introduced in 2.0, then you can start by optimizing your model with one line: `model = torch.compile(model)`.  
While the speedups are primarily observed during training, you can also use it for inference if your model runs faster than eager mode.

    ```python
    import torch
      
    def train(model, dataloader):
      model = torch.compile(model)
      for batch in dataloader:
        run_epoch(model, batch)

    def infer(model, input):
      model = torch.compile(model)
      return model(\*\*input)
    ```

7. **Why should I use PT2.0 instead of PT 1.X?**  
See answer to Question (2).

8. **What is my code doing differently when running PyTorch 2.0?**  
Out of the box, PyTorch 2.0 is the same as PyTorch 1.x, your models run in eager-mode i.e. every line of Python is executed one after the other.  
In 2.0, if you wrap your model in `model = torch.compile(model)`, your model goes through 3 steps before execution:  
    1. Graph acquisition: first the model is rewritten as blocks of subgraphs. Subgraphs which can be compiled by TorchDynamo are “flattened” and the other subgraphs (which might contain control-flow code or other unsupported Python constructs) will fall back to Eager-Mode.  
    2. Graph lowering: all the PyTorch operations are decomposed into their constituent kernels specific to the chosen backend.  
    3. Graph compilation, where the kernels call their corresponding low-level device-specific operations.  

9. **What new components does PT2.0 add to PT?**  
    - **TorchDynamo** generates FX Graphs from Python bytecode. It maintains the eager-mode capabilities using [guards](https://pytorch.org/docs/stable/torch.compiler_guards_overview.html#caching-and-guards-overview) to ensure the generated graphs are valid ([read more](https://dev-discuss.pytorch.org/t/torchdynamo-an-experiment-in-dynamic-python-bytecode-transformation/361))  
    - **AOTAutograd** to generate the backward graph corresponding to the forward graph captured by TorchDynamo ([read more](https://dev-discuss.pytorch.org/t/torchdynamo-update-6-training-support-with-aotautograd/570)).  
    - **PrimTorch** to decompose complicated PyTorch operations into simpler and more elementary ops ([read more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-2/645)).  
    - **\[Backend]** Backends integrate with TorchDynamo to compile the graph into IR that can run on accelerators. For example, **TorchInductor** compiles the graph to either **Triton** for GPU execution or **OpenMP** for CPU execution ([read more](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)).  
  
10. **What compiler backends does 2.0 currently support?**  
The default and the most complete backend is [TorchInductor](https://github.com/pytorch/pytorch/tree/master/torch/_inductor), but TorchDynamo has a growing list of backends that can be found by calling `torchdynamo.list_backends()`.  
  
11. **How does distributed training work with 2.0?**  
DDP and FSDP in Compiled mode  can run up to 15% faster than Eager-Mode in FP32 and up to 80% faster in AMP precision. PT2.0 does some extra optimization to ensure DDP’s communication-computation overlap works well with Dynamo’s partial graph creation. Ensure you run DDP with static_graph=False. More details [here](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860).  

12. **How can I learn more about PT2.0 developments?**  
The [PyTorch Developers forum](http://dev-discuss.pytorch.org/) is the best place to learn about 2.0 components directly from the developers who build them.  

13. **Help my code is running slower with 2.0’s Compiled Mode!**  
The most likely reason for performance hits is too many graph breaks. For instance, something innocuous as a print statement in your model’s forward triggers a graph break. We have ways to diagnose these  - read more [here](https://pytorch.org/docs/stable/torch.compiler_faq.html#why-am-i-not-seeing-speedups).  

14. **My previously-running code is crashing with 2.0’s Compiled Mode! How do I debug it?**  
Here are some techniques to triage where your code might be failing, and printing helpful logs: [https://pytorch.org/docs/stable/torch.compiler_faq.html#why-is-my-code-crashing](https://pytorch.org/docs/stable/torch.compiler_faq.html#why-is-my-code-crashing).  

## Ask the Engineers: 2.0 Live Q&A Series

We will be hosting a series of live Q&A sessions for the community to have deeper questions and dialogue with the experts. Please check back to see the full calendar of topics throughout the year. If you are unable to attend: 1) They will be recorded for future viewing and 2) You can attend our Dev Infra Office Hours every Friday at 10 AM PST @ [https://github.com/pytorch/pytorch/wiki/Dev-Infra-Office-Hours](https://github.com/pytorch/pytorch/wiki/Dev-Infra-Office-Hours).

Please click [here](https://pytorchconference22.splashthat.com/) to see dates, times, descriptions and links.  

Disclaimer: Please do not share your personal information, last name, company when joining the live sessions and submitting questions.  

<table style="min-width: 350px" class="QnATable">
  <tr>
   <td style="width:50%"><b>TOPIC</b></td>
   <td style="width:50%"><b>HOST</b></td>
  </tr>
  <tr>
   <td><b>The new developer experience of using 2.0 (install, setup, clone an example, run with 2.0)</b></td>
   <td>Suraj Subramanian<br>
   <a href="https://www.linkedin.com/in/surajsubramanian/">LinkedIn</a> |
   <a href="https://twitter.com/subramen">Twitter</a>
   </td>
  </tr>
  <tr>
   <td><a href="https://www.youtube.com/watch?v=1FSBurHpH_Q&list=PL_lsbAsL_o2CQr8oh5sNWt96yWQphNEzM&index=2"><b>PT2 Profiling and Debugging</b></a></td>
   <td>Bert Maher<br>
   <a href="https://www.linkedin.com/in/bertrand-maher/">LinkedIn</a> |
   <a href="https://twitter.com/tensorbert">Twitter</a>
   </td>
  </tr>
  <tr>
   <td><a href="https://community.linuxfoundation.org/j/gayr75zshnded/"><b>A deep dive on TorchInductor and PT 2.0 Backend Integration</b></a></td>
   <td>Natalia Gimelshein, Bin Bao and Sherlock Huang<br>
   Natalia Gimelshein<br>
   <a href="https://www.linkedin.com/in/natalia-gimelshein-8347a480/">LinkedIn</a><br>
   Sherlock Huang<br>
   <a href="https://www.linkedin.com/in/sherlock-baihan-huang-07787a59/">LinkedIn</a>
   </td>
  </tr>
  <tr>
   <td><b>Extend PyTorch without C++ and functorch: JAX-like composable function transforms for PyTorch</b></td>
   <td>Anjali Chourdia and Samantha Andow<br>
   Anjali Chourdia<br>
   <a href="https://www.linkedin.com/in/anjali-chourdia/">LinkedIn</a> |
   <a href="https://twitter.com/AChourdia">Twitter</a><br>
   Samantha Andow<br>
   <a href="https://www.linkedin.com/in/samantha-andow-1b6965a7/">LinkedIn</a> |
   <a href="https://twitter.com/_samdow">Twitter</a>
   </td>
  </tr>
  <tr>
   <td><a href="https://www.youtube.com/watch?v=5FNHwPIyHr8&list=PL_lsbAsL_o2CQr8oh5sNWt96yWQphNEzM&index=3"><b>A deep dive on TorchDynamo</b></a></td>
   <td>Michael Voznesensky<br>
   <a href="https://www.linkedin.com/in/michael-voznesensky-70459624/">LinkedIn</a>
   </td>
  </tr>
  <tr>
   <td><b>Rethinking data loading with TorchData:Datapipes and Dataloader2</b></td>
   <td>Kevin Tse<br>
   <a href="https://www.linkedin.com/in/kevin-tse-35051367/">LinkedIn</a>
   </td>
  </tr>
  <tr>
   <td><b>Composable training (+ torcheval, torchsnapshot)</b></td>
   <td>Ananth Subramaniam</td>
  </tr>
  <tr>
   <td><a href="https://www.youtube.com/watch?v=v4nDZTK_eJg&list=PL_lsbAsL_o2CQr8oh5sNWt96yWQphNEzM&index=1"><b>How and why contribute code and tutorials to PyTorch</b></a></td>
   <td>Zain Rizvi, Svetlana Karslioglu and Carl Parker<br>
   Zain Rizvi<br>
   <a href="https://linkedin.com/in/zainrizvi">LinkedIn</a> |
   <a href="https://twitter.com/zainrzv">Twitter</a><br>
   Svetlana Karslioglu<br>
   <a href="https://www.linkedin.com/in/svetlana-karslioglu">LinkedIn</a> |
   <a href="https://twitter.com/laignas">Twitter</a>
   </td>
  </tr>
  <tr>
   <td><b>Dynamic Shapes and Calculating Maximum Batch Size</b></td>
   <td>Edward Yang and Elias Ellison<br>
   Edward Yang<br>
   <a href="https://twitter.com/ezyang">Twitter</a>
   </td>
  </tr>
  <tr>
   <td><a href="https://www.youtube.com/watch?v=U6J5hl6nXlU&list=PL_lsbAsL_o2CQr8oh5sNWt96yWQphNEzM&index=4"><b>PyTorch 2.0 Export: Sound Whole Graph Capture for PyTorch</b></a></td>
   <td>Michael Suo and Yanan Cao<br>
   Yanan Cao<br>
   <a href="https://www.linkedin.com/in/yanan-cao-65836020/">LinkedIn</a>
   </td>
  </tr>
  <tr>
   <td><b>2-D Parallelism using DistributedTensor and PyTorch DistributedTensor</b></td>
   <td>Wanchao Liang and Alisson Gusatti Azzolini<br>
   Wanchao Liang<br>
   <a href="https://www.linkedin.com/in/wanchaol/">LinkedIn</a> |
   <a href="https://twitter.com/wanchao_">Twitter</a><br>
   Alisson Gusatti Azzolini<br>
   <a href="https://www.linkedin.com/in/alissonazzolini/">LinkedIn</a>
   </td>
  </tr>
  <tr>
   <td><a href="https://www.youtube.com/watch?v=NgW6gp69ssc&list=PL_lsbAsL_o2CQr8oh5sNWt96yWQphNEzM&index=5"><b>TorchRec and FSDP in Production</b></a></td>
   <td>Dennis van der Staay, Andrew Gu and Rohan Varma<br>
   Dennis van der Staay<br>
   <a href="https://www.linkedin.com/in/staay/">LinkedIn</a><br>
   Rohan Varma<br>
   <a href="https://www.linkedin.com/in/varmarohan/">LinkedIn</a> |
   <a href="https://twitter.com/rvarm1">Twitter</a>
   </td>
  </tr>
  <tr>
   <td><b>The Future of PyTorch On-Device</b></td>
   <td>Raziel Alvarez Guevara<br>
   <a href="https://www.linkedin.com/in/razielalvarez/">LinkedIn</a> |
   <a href="https://twitter.com/razielag">Twitter</a>
   </td>
  </tr>
  <tr>
   <td><b>TorchMultiModal</b><br>
   <a href="https://pytorch.org/blog/introducing-torchmultimodal/" target="_blank">Intro Blog</a><br>
   <a href="https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/" target="_blank">Scaling Blog</a></td>
   <td>Kartikay Khandelwal<br>
   <a href="https://www.linkedin.com/in/kartikaykhandelwal/">LinkedIn</a> |
   <a href="https://twitter.com/kakemeister">Twitter</a>
   </td>
  </tr>
  <tr>
   <td><b>BetterTransformers (+ integration with Hugging Face), Model Serving and Optimizations</b><br>
   <a href="https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2" target="_blank">Blog 1</a><br>
   <a href="https://github.com/pytorch/serve" target="_blank">Github</a></td>
   <td>Hamid Shojanazeri and Mark Saroufim<br>
   Mark Saroufim<br>
   <a href="https://www.linkedin.com/in/marksaroufim/">LinkedIn</a> |
   <a href="https://twitter.com/marksaroufim">Twitter</a>
   </td>
  </tr>
  <tr>
   <td><a href="https://community.linuxfoundation.org/j/5s25r7uxmpq5e/"><b>PT2 and Distributed</b></a></td>
   <td>Will Constable<br>
   <a href="https://www.linkedin.com/in/will-constable-969a53b/">LinkedIn</a>
   </td>
  </tr>
</table>

## Watch the Talks from PyTorch Conference

- [TorchDynamo](https://www.youtube.com/watch?v=vbtGZL7IrAw)
- [TorchInductor](https://www.youtube.com/watch?v=vbtGZL7IrAw)
- [Dynamic Shapes](https://www.youtube.com/watch?v=vbtGZL7IrAw)
- [Export Path](https://www.youtube.com/watch?v=vbtGZL7IrAw)

<script src="{{ site.baseurl }}/assets/get-started-sidebar.js"></script>
