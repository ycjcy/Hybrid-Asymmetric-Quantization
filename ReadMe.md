# Hybrid-Asymmetric-Quantization

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

## Table of Contents

- [Introduction](#Introduction)
- [Implement](#Implement)
- [Experiment](#Experiment)
- [License](#license)

## Introduction 


In this paper "Activation Redistribution based Hybrid Asymmetric Quantization Method of Neural Networks", an efficient adaptive hybrid asymmetric quantization method for different types of neural network layers is proposed. The proposed method can resolve the contradiction between the quantization accuracy and the ease of implementation, and balance the trade-off between clipping range and quantization resolution, and thus improve the accuracy of the quantized neural network. 

Activation Redistribution based Hybrid Asymmetric Quantization Method of Neural Networks for Integer-Only Inference Results on PC devices.

The purpose of the experiments are to verify the effectiveness of the proposed hybrid asymmetric Integer-only quantization method. 

## Implement

Result.py for Execute Script (Moving MinMax、MinMax、Histogram、Proposed Activation redistribution), to Reproduce experimental results.

```sh
$ python result.py method_param
```
    method_param is optional param.
        -A Moving MinMax 
        -B MinMax
        -C Histogram
        -D Proposed Activation redistribution)

## Experiment
Experimental Results of the Proposed Hybrid Asymmetric Quantization Method and Pytorch for Classification Application on fake-quantization software.

    The dataset for image classification application is ImageNet. ImageNet is an image database organized according to the WordNet hierarchy, in which each node of the hierarchy is depicted by hundreds and thousands of images. The dataset has been instrumental in advancing computer vision and deep learning research.

| Model | PC_Accuracy(FP32) | PyTorch_Accuracy_(fake_INT8) |  | Proposed_Quantization_Accuracy(fake INT 8) |  |  |
| ------ | ------ | ------ | ------ | ------ | ------ | ------|
|  |  |  | Moving MinMax  | MinMax | Histogram | Proposed Activation Redistribution |
| GoogleNet | 67.04% | 66.50% | 66.82% | 66.82% | 66.77% | 66.85% |
| MobileNetV2 |	70.24% | 66.82% | 66.60% | 66.23% |	66.99% | 67.38% |
| VGG16 | 66.13% | 65.22% | 66.20%	| 65.12% | 65.31% |	66.26% |

## License

[Xi’an Microelectronics Technology Institute](LICENSE) © ycjcy
