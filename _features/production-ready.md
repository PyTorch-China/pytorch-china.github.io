---
title: 生产就绪
order: 1
snippet: >
  ```python
    import torch
    class MyModule(torch.nn.Module):

      def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

      def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output

      # Compile the model code to a static representation
      my_script_module = torch.jit.script(MyModule(3, 4))

      # Save the compiled code and model data so it can be loaded elsewhere
      my_script_module.save("my_script_module.pt")
  ```

# summary-home: Transition seamlessly between eager and graph modes with TorchScript, and accelerate the path to production with TorchServe.
summary-home: 使用 TorchScript 无缝过渡到急切模式和图模式之间，并借助 TorchServe 加速进入生产环境的路径。
featured-home: true

---

<!-- With TorchScript, PyTorch provides ease-of-use and flexibility in eager mode, while seamlessly transitioning to graph mode for speed, optimization, and functionality in C++ runtime environments. -->

通过 TorchScript，PyTorch 在急切模式下提供了易用性和灵活性，同时在 C++ 运行时环境中无缝过渡到图模式，以提高速度、优化和功能。
