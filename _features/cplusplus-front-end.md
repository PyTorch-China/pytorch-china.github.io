---
# title: C++ Front-End
title: C++ 前端
order: 7
snippet: >
  ```cpp
    #include <torch/torch.h>

    torch::nn::Linear model(num_features, 1);
    torch::optim::SGD optimizer(model->parameters());
    auto data_loader = torch::data::data_loader(dataset);

    for (size_t epoch = 0; epoch < 10; ++epoch) {
      for (auto batch : data_loader) {
        auto prediction = model->forward(batch.data);
        auto loss = loss_function(prediction, batch.target);
        loss.backward();
        optimizer.step();
      }
    }
  ```
---

<!-- The C++ frontend is a pure C++ interface to PyTorch that follows the design and architecture of the established Python frontend. It is intended to enable research in high performance, low latency and bare metal C++ applications. -->

C++ 前端是 PyTorch 的纯 C++ 接口，遵循已建立的 Python 前端的设计和架构。它旨在实现在高性能、低延迟和裸机 C++ 应用中的研究。
