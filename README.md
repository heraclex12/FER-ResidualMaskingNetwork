# Residual Masking Network
The state of the art Residual Masking Network model in Keras for Facial Expression Recognition

This is my school assignment project (Pattern Recognition) which was implemented using Keras (Tensorflow backend)

## Installation
This prerequisites are Keras 2.3.1 and python3
```
> pip install requirements.txt
```

## Benchmarking on FER2013

We benchmark our code thoroughly on dataset: FER2013. Below are the results and trained weights:


Model | Accuracy |
---------|--------|
ResNet50 | 66%
ResNet50 with VGGFace-pretrained | 69.23%
ResNet34 with pretrained-model | 69.27%
ResMaskingNet (Ours) | 71.14%
ResMaskingNet \[1\] (from paper) | 74.14




## Reference
[1][Luan Pham & Tuan Anh Tran. Facial Expression Recognition using Residual Masking Network 2020](https://github.com/phamquiluan/ResidualMaskingNetwork)
