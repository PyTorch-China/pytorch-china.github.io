---
# title: Cloud Support
title: 云支持
order: 8
snippet: >
  ```sh
    export IMAGE_FAMILY="pytorch-latest-cpu"
    export ZONE="us-west1-b"
    export INSTANCE_NAME="my-instance"
    
    gcloud compute instances create $INSTANCE_NAME \
      --zone=$ZONE \
      --image-family=$IMAGE_FAMILY \
      --image-project=deeplearning-platform-release
  ```

# summary-home: PyTorch is well supported on major cloud platforms, providing frictionless development and easy scaling.
summary-home: PyTorch 在主要的云平台上得到了良好的支持，提供了高效的开发和简便的扩展。
featured-home: true

---

<!-- PyTorch is well supported on major cloud platforms, providing frictionless development and easy scaling through prebuilt images, large scale training on GPUs, ability to run models in a production scale environment, and more. -->

PyTorch 在主要的云平台上得到了良好的支持，提供了无缝开发和简便扩展的功能，包括预构建的镜像、大规模的 GPU 训练、在生产环境中运行模型的能力等。
