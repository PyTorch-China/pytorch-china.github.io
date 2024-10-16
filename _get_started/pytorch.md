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

## PyTorch 2.x：更快，更 Python 化，并且一如既往地支持动态

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

## 用户评价

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
- 图降级
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

### 模式

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

通过在 PyTorch 2.0 的编译模式中支持动态形状，我们可以获得性能和易用性的最佳结合。

<div style="display:flex; flex-direction: row; padding: 10px;">
<img src="/assets/images/pytorch-2.0-img7.png" width="50%">
<img src="/assets/images/pytorch-2.0-img8.png" width="50%">
</div>

当前的工作进展非常迅速，我们可能会暂时让一些模型回归，因为我们正在进行基础设施的重大改进。关于动态形状的最新进展可以在 [这里](https://dev-discuss.pytorch.org/t/state-of-symbolic-shapes-branch/777/19)找到。

## 分布式

torch.distributed 的两个主要分布式包装器在编译模式下工作良好。

`DistributedDataParallel` (DDP) 和 `FullyShardedDataParallel` (FSDP) 都能在编译模式下工作，并且相对于 eager 模式提供了更好的性能和内存利用率，但也有一些注意事项和限制。


<p>
<center> <u>AMP 精度下的加速</u></center>
<img src="/assets/images/pytorch-2.0-img9.png" width="90%">
<center><u>左图：编译模式下 FSDP 相对于 eager 模式的加速（AMP 精度）。<br>
右图：编译模式下 FSDP 占用的内存明显少于 eager 模式</u></center>
</p>

<div style="display:flex; flex-direction: row; padding:10px;">
<img src="/assets/images/pytorch-2.0-img10.png" width="50%">
<img src="/assets/images/pytorch-2.0-img11.png" width="50%">
</div>

### DistributedDataParallel (DDP)

DDP 依赖于与反向计算重叠的 AllReduce 通信，并将较小的每层 AllReduce 操作分组到 'buckets' 中以提高效率。由 TorchDynamo 编译的 AOTAutograd 函数在与 DDP 组合时会阻止通信重叠，但通过为每个 ‘buckets’ 编译单独的子图并允许通信操作在子图外部和子图之间发生，可以恢复性能。编译模式下的 DDP 支持目前还需要 `static_graph=False`。有关 DDP + TorchDynamo 方法和结果的更多详细信息，请参见 [这篇文章](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)。


### FullyShardedDataParallel (FSDP)

FSDP 本身是一个 “beta” PyTorch 功能，由于能够调整包装的子模块以及通常有更多的配置选项，因此其系统复杂性高于 DDP。如果配置了 `use_original_params=True` 标志，FSDP 可以与 TorchDynamo 和 TorchInductor 一起用于各种主流模型。目前预期会有一些与特定模型或配置的兼容性问题，但这些问题将会得到积极改进，如果在 GitHub 上提交问题，特定模型可以优先处理。

用户指定一个 `auto_wrap_policy` 参数，以指示其模型的哪些子模块一起包装在一个用于状态分片的 FSDP 实例中，或者手动将子模块包装在 FSDP 实例中。例如，许多 transformer 模型在每个 “transformer block” 包装在一个单独的 FSDP 实例中时效果很好，因此一次只需要实现一个 transformer 块的完整状态。Dynamo 将在每个 FSDP 实例的边界插入图中断，以允许前向（和后向）中的通信操作在图外部并行于计算发生。

如果 FSDP 在不将子模块包装在单独实例中的情况下使用，它将回退到类似于 DDP 的操作，但没有分桶。因此，所有梯度在一个操作中减少，并且即使在 eager 模式下也无法进行计算/通信重叠。此配置仅在功能上与 TorchDynamo 进行了测试，但未进行性能测试。

## 开发者体验

在 PyTorch 2.0 中，我们希望简化后端（编译器）集成体验。为此，我们专注于 **减少操作符的数量** 和 **简化操作符集的语义**，以便于开发接入 PyTorch 后端。

在图形形式中，PT2 堆栈如下所示：

<p>
<img src="/assets/images/pytorch-2.0-img12.png" width="90%">
</p>

从图的中间开始，AOTAutograd 以提前捕获的方式动态捕获自动微分逻辑，生成前向和后向操作符的 FX 图。

我们提供了一组强化的分解（即用其他操作符实现的操作符实现），可以用来减少后端需要实现的操作符数量。我们还通过一个称为 _functionalization_ 的过程选择性地重写复杂的 PyTorch 逻辑，包括变换和视图，以简化 PyTorch 操作符的语义，并保证操作符元数据信息，如形状传播公式。这项工作正在积极进行中；我们的目标是提供一组 _原始_ 且 _稳定_ 的约 250 个操作符，称为 _PrimTorch_，供应商可以利用（即选择加入）以简化其集成。

在减少和简化操作符集之后，后端可以选择在 Dynamo（即中间层，紧接在 AOTAutograd 之后）或 Inductor（较低层）进行集成。我们在下面描述了一些做出此选择时的考虑因素，以及围绕混合后端的未来工作。

**Dynamo 后端**

拥有现有编译器栈的供应商可能会发现，将其集成为 TorchDynamo 后端最为容易，接收 ATen/Prims IR 形式的 FX 图。请注意，对于训练和推理，集成点将紧接在 AOTAutograd 之后，因为我们目前将分解作为 AOTAutograd 的一部分，并且如果目标是推理，则仅跳过特定于反向传播的步骤。


**Inductor 后端**

供应商也可以将其后端直接集成到 Inductor 中。Inductor 接收由 AOTAutograd 生成的包含 ATen/Prim 操作的图，并进一步将其降低到循环级别的 IR。如今，Inductor 为点操作、归约、散射/聚集和窗口操作提供了循环级别 IR 的降低。此外，Inductor 创建融合组，进行索引简化，维度折叠，并调整循环迭代顺序以支持高效的代码生成。供应商可以通过提供从循环级别 IR 到硬件特定代码的映射来进行集成。目前，Inductor 有两个后端：(1) 生成多线程 CPU 代码的 C++，(2) 生成高性能 GPU 代码的 Triton。这些 Inductor 后端可以作为替代后端的灵感来源。

**混合后端接口（即将推出）**

我们已经构建了用于将 FX 图划分为包含后端支持的操作符的子图并 eager 执行其余部分的实用程序。这些实用程序可以扩展以支持 “混合后端”，配置图的哪些部分运行于哪个后端。然而，目前还没有稳定的接口或合同供后端公开其操作符支持、操作符模式偏好等。这仍然是正在进行的工作，我们欢迎早期试用者的反馈。

## 最终想法

我们对PyTorch 2.0及其未来的发展方向感到非常兴奋。通往2.0最终版本的道路将会很艰难，请早日加入我们的旅程。如果您有兴趣深入研究或为编译器做贡献，请继续阅读下面的内容，其中包括如何入门的更多信息（例如，教程、基准测试、模型、常见问题解答）以及本月开始的工程师问答：2.0直播问答系列。其他资源包括：

- [入门指南](https://pytorch.org/docs/stable/torch.compiler_get_started.html)
- [教材](https://pytorch.org/tutorials/)
- [文档](https://pytorch.org/docs/stable)
- [开发者论坛](https://dev-discuss.pytorch.org)

<script page-id="pytorch" src="{{ site.baseurl }}/assets/menu-tab-selection.js"></script>
<script src="{{ site.baseurl }}/assets/quick-start-module.js"></script>
<script src="{{ site.baseurl }}/assets/show-screencast.js"></script>

## 使用PyTorch 2.0加速Hugging Face和TIMM模型

作者: Mark Saroufim

通过使用单行装饰器 `torch.compile()` 实验不同的编译器后端以加速PyTorch代码变得非常容易。它可以直接用于 nn.Module，作为 torch.jit.script() 的替代品，但不需要进行任何源代码更改。我们预计这一行代码的更改将为您已经运行的大多数模型提供 30%-2 倍的训练时间加速。

```python
opt_module = torch.compile(module)
```

torch.compile 支持任意的 PyTorch 代码、控制流、变换，并且还支持动态形状。我们对这一发展感到非常兴奋，因此称其为PyTorch 2.0。



这次公告的不同之处在于，我们已经对一些最受欢迎的开源PyTorch模型进行了基准测试，并获得了从30%到2倍的显著加速 [https://github.com/pytorch/torchdynamo/issues/681](https://github.com/pytorch/torchdynamo/issues/681).

这里没有任何技巧，我们已经pip安装了流行的库，如 [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers), [https://github.com/huggingface/accelerate](https://github.com/huggingface/accelerate) and [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)，然后运行torch.compile()，就是这样简单。

同时获得性能和便利性是很罕见的，这也是核心团队对 PyTorch 2.0 感到如此兴奋的原因。

## 要求

对于GPU（新一代GPU将看到显著更好的性能）

```
pip3 install numpy --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
```

对于CPU

```
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

可选：验证安装

```
git clone https://github.com/pytorch/pytorch
cd tools/dynamo
python verify_dynamo.py
```

可选：Docker安装

我们还提供了所有所需依赖项的PyTorch夜间版本二进制文件，您可以通过以下方式下载


```
docker pull ghcr.io/pytorch/pytorch-nightly
```

对于临时实验，只需确保您的容器可以访问所有GPU

```
docker run --gpus all -it ghcr.io/pytorch/pytorch-nightly:latest /bin/bash
```

## 入门指南

请阅读Mark Saroufim的 [完整博客](/blog/Accelerating-Hugging-Face-and-TIMM-models/)，他会引导您完成教程和使用真实模型来尝试 PyTorch 2.0。

我们开发PyTorch的目标是构建一个广度优先的编译器，以加速开源中人们实际运行的大多数模型。Hugging Face Hub 最终成为了一个非常有价值的基准测试工具，确保我们进行的任何优化实际上都有助于加速人们想要运行的模型。

博客教程将准确展示如何复制这些加速效果，以便您能像我们一样对 PyTorch 2.0 感到兴奋。所以请尝试 PyTorch 2.0，享受免费的性能提升，如果您没有看到效果，请提交一个issue给我们，我们将确保您的模型得到支持 [https://github.com/pytorch/torchdynamo/issues](https://github.com/pytorch/torchdynamo/issues)

毕竟，除非 **您的模型** 实际运行得更快，否则我们不能声称我们创建了一个广度优先的编译器。

## FAQs  

1. **什么是PT 2.0?**  
2.0 是最新的 PyTorch 版本。PyTorch 2.0 提供了相同的 eager-mode 开发体验，同时通过 torch.compile 添加了编译模式。此编译模式有可能在训练和推理期间加速您的模型。

2. **为什么是 2.0 而不是 1.14?**  
PyTorch 2.0 是 1.14 的升级版。我们发布了大量新功能，这些功能改变了您使用 PyTorch 的方式，因此我们称之为 2.0。

3. **如何安装 2.0？有任何额外要求吗？**

    安装最新的 nightlies:

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

4. **2.0 代码与 1.X 兼容吗？**  
是的，使用 2.0 不需要修改您的 PyTorch 工作流程。只需一行代码 `model = torch.compile(model)` 就可以优化您的模型以使用 2.0 堆栈，并与您的其他 PyTorch 代码顺利运行。这是完全可选的，您不需要必须使用新的编译器。

5. **2.0 默认启用吗?**  
2.0 是发布的名称。torch.compile 是 2.0 中发布的功能，您需要显式使用 torch.compile。

6. **如何将我的 PT1.X 代码迁移到 PT2.0?**  
您的代码应该可以按原样工作，无需任何迁移。如果您想使用 2.0 中引入的新编译模式功能，可以通过一行代码开始优化您的模型： `model = torch.compile(model)`。

虽然主要在训练期间观察到加速，但如果您的模型比 eager 模式运行得更快，您也可以在推理中使用它。


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

7. **为什么我应该使用 PT2.0 而不是 PT 1.X?**  
请参阅问题 (2) 的答案。

8. **在运行 PyTorch 2.0 时，我的代码有什么不同?**  
开箱即用，PyTorch 2.0 与 PyTorch 1.x 相同，您的模型在 eager-mode 下运行，即每行 Python 代码依次执行。
在 2.0 中，如果您将模型包装在 model = torch.compile(model) 中，您的模型在执行前会经过 3 个步骤：

    1. 图获取：首先将模型重写为子图块。可以由 TorchDynamo 编译的子图将被“展平”，而其他子图（可能包含控制流代码或其他不支持的 Python 构造）将回退到 Eager-Mode。
    2. 图降级：所有 PyTorch 操作都分解为特定于所选后端的基本内核。
    3. 图编译：内核调用其相应的低级设备特定操作。

9. **PT2.0 为 PT 添加了哪些新组件?**  
    - **TorchDynamo** 从 Python 字节码生成 FX 图。它使用 [guards](https://pytorch.org/docs/stable/torch.compiler_guards_overview.html#caching-and-guards-overview) 保持 eager-mode 功能，以确保生成的图是有效的 ([了解更多](https://dev-discuss.pytorch.org/t/torchdynamo-an-experiment-in-dynamic-python-bytecode-transformation/361))  
    - **AOTAutograd** 生成与 TorchDynamo 捕获的前向图对应的后向图 ([了解更多](https://dev-discuss.pytorch.org/t/torchdynamo-update-6-training-support-with-aotautograd/570)).  
    - **PrimTorch** 将复杂的 PyTorch 操作分解为更简单和更基本的操作 ([了解更多](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-2/645)).  
    - **\[Backend]** 后端与 TorchDynamo 集成，将图编译为可以在加速器上运行的 IR。例如，**TorchInductor** 将图编译为 **Triton** 以进行 GPU 执行或 **OpenMP** 以进行 CPU 执行 ([了解更多](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)).  
  
10. **2.0 目前支持哪些编译器后端?**  
默认且最完整的后端是 [TorchInductor](https://github.com/pytorch/pytorch/tree/master/torch/_inductor)，但 TorchDynamo 有一个不断增长的后端列表，可以通过调用 `torchdynamo.list_backends()` 找到。

11. **2.0 的分布式训练如何工作?**  
在编译模式下，DDP 和 FSDP 在 FP32 中可以比 Eager-Mode 快 15%，在 AMP 精度中可以快 80%。PT2.0 进行了一些额外的优化，以确保 DDP 的通信计算重叠与 Dynamo 的部分图创建配合良好。确保您在运行 DDP 时将 static_graph 设置为 False。更多详情请参阅 [这里](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)。

12. **如何了解更多关于 PT2.0?**  
[PyTorch Developers forum](http://dev-discuss.pytorch.org/) 是了解 2.0 组件的最佳地方，您可以直接从构建它们的开发者那里获取信息。

13. **我的代码在 2.0 的编译模式下运行得更慢!**  
性能下降的最可能原因是图中断过多。例如，模型前向中的一个无害的 print 语句会触发图中断。我们有方法诊断这些问题 - 请阅读 [这里](https://pytorch.org/docs/stable/torch.compiler_faq.html#why-am-i-not-seeing-speedups)。

14. **我的代码在 2.0 的编译模式下崩溃了！如何调试?**  
这里有一些技术可以帮助您排查代码可能失败的地方，并打印有用的日志：[https://pytorch.org/docs/stable/torch.compiler_faq.html#why-is-my-code-crashing](https://pytorch.org/docs/stable/torch.compiler_faq.html#why-is-my-code-crashing)

## Ask the Engineers: 2.0 Live Q&A Series

我们将举办一系列的现场问答环节，让社区能够与专家进行更深入的问题和对话。请随时查看全年主题的完整日历。如果您无法参加：1) 这些活动将被录制以供日后观看，2) 您可以参加我们每周五上午10点（太平洋标准时间）的开发基础设施办公时间 @ [https://github.com/pytorch/pytorch/wiki/Dev-Infra-Office-Hours](https://github.com/pytorch/pytorch/wiki/Dev-Infra-Office-Hours)。

请点击 [这里](https://pytorchconference22.splashthat.com/) 查看日期、时间、描述和链接。

免责声明：请不要在加入现场会议和提交问题时分享您的个人信息、姓氏、公司。

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

## PyTorch Conference 相关演讲

- [TorchDynamo](https://www.youtube.com/watch?v=vbtGZL7IrAw)
- [TorchInductor](https://www.youtube.com/watch?v=vbtGZL7IrAw)
- [Dynamic Shapes](https://www.youtube.com/watch?v=vbtGZL7IrAw)
- [Export Path](https://www.youtube.com/watch?v=vbtGZL7IrAw)

<script src="{{ site.baseurl }}/assets/get-started-sidebar.js"></script>
