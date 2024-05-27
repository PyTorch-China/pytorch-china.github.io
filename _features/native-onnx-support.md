---
# title: Native ONNX Support
title: 原生 ONNX 支持
order: 6
snippet: >
  ```python
    import torch.onnx
    import torchvision

    dummy_input = torch.randn(1, 3, 224, 224)
    model = torchvision.models.alexnet(pretrained=True)
    torch.onnx.export(model, dummy_input, "alexnet.onnx")
  ```
---

<!-- Export models in the standard ONNX (Open Neural Network Exchange) format for direct access to ONNX-compatible platforms, runtimes, visualizers, and more. -->

将模型导出为标准的 ONNX（开放神经网络交换）格式，以便直接在兼容 ONNX 平台、运行时、可视化工具等上访问。
