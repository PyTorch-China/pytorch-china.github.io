---
# title: Mobile (Experimental)
title: 移动端 (实验阶段)
order: 4
snippet: >
  ```python
    ## Save your model
    torch.jit.script(model).save("my_mobile_model.pt")

    ## iOS prebuilt binary
    pod ‘LibTorch’
    ## Android prebuilt binary
    implementation 'org.pytorch:pytorch_android:1.3.0'

    ## Run your model (Android example)
    Tensor input = Tensor.fromBlob(data, new long[]{1, data.length});
    IValue output = module.forward(IValue.tensor(input));
    float[] scores = output.getTensor().getDataAsFloatArray();
  ```

# summary-home: PyTorch supports an end-to-end workflow from Python to deployment on iOS and Android. It extends the PyTorch API to cover common preprocessing and integration tasks needed for incorporating ML in mobile applications.
summary-home: PyTorch 支持从 Python 到在 iOS 和 Android 上部署的端到端工作流程。它扩展了 PyTorch API，以涵盖在移动应用程序中集成机器学习所需的常见预处理和集成任务。
featured-home: false

---

<!-- PyTorch supports an end-to-end workflow from Python to deployment on iOS and Android. It extends the PyTorch API to cover common preprocessing and integration tasks needed for incorporating ML in mobile applications. -->

PyTorch 支持从 Python 到在 iOS 和 Android 上部署的端到端工作流程。它扩展了 PyTorch API，以涵盖在移动应用程序中集成 ML 所需的常见预处理和集成任务。
