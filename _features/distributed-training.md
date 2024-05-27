---
# title: Distributed Training
title: 分布式训练
order: 3
snippet: >
  ```python
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    
    dist.init_process_group(backend='gloo')
    model = DistributedDataParallel(model)
  ```

# summary-home: Scalable distributed training and performance optimization in research and production is enabled by the torch.distributed backend.
summary-home: torch.distributed 后端支持可扩展的分布式训练和性能优化，在研究和生产中发挥作用。
featured-home: true

---

<!-- Optimize performance in both research and production by taking advantage of native support for asynchronous execution of collective operations and peer-to-peer communication that is accessible from Python and C++. -->

通过利用原生支持的异步执行集合操作和点对点通信，从而优化研究和生产中的性能，在 Python 和 C++ 中都可以访问。
