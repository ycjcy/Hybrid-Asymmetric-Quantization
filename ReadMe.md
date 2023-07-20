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
Classification Application

Experimental Results of the Proposed Hybrid Asymmetric Quantization Method and Pytorch for Classification Application on fake-quantization software.

    The dataset for image classification application is ImageNet(https://image-net.org/update-mar-11-2021.php). ImageNet is an image database organized according to the WordNet hierarchy, in which each node of the hierarchy is depicted by hundreds and thousands of images. The dataset has been instrumental in advancing computer vision and deep learning research.
    The results for the classification application are shown in the table and figure below. For the three classification models, the accuracy of the proposed hybrid asymmetric quantization method (66.85%, 67.38%, 66.26%) is the highest compared with PyTorch (66.50%, 66.82%, 65.22%) and NNI (66.74%, 66.50%, 65.20%). At the same time, there are three ways in PyTorch and NNI to compute the clipping range. For different models, the best strategy of PyTorch and NNI to compute the clipping range is different. The proposed activation redistribution method outperforms the three strategies of PyTorch and NNI.

| Model | PC_Accuracy(FP32) | NNI Accuracy (fake INT8 ) | PyTorch_Accuracy_(fake_INT8) |  | Proposed_Quantization_Accuracy(fake INT 8) |  |  |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------|
|  |  |  | Moving MinMax  | MinMax | Histogram | Proposed Activation Redistribution |
| GoogleNet | 67.04% | 66.74% | 66.50% | 66.82% | 66.82% | 66.77% | 66.85% |
| MobileNetV2 |	70.24% | 66.50% | 66.82% | 66.60% | 66.23% |	66.99% | 67.38% |
| VGG16 | 66.13% | 65.20% | 65.22% | 66.20%	| 65.12% | 65.31% |	66.26% |

Small Target Detection Application

The model used for small target detection to compare with PyTorch is the same as section 4.5.1. The evaluation metric is mAP. We compare the proposed hybrid asymmetric quantization method with PyTorch, and compare the proposed activation redistribution method with MinMax, MovingAverage, and Histogram in PyTorch on fake-quantization software.
The dataset for is small target detection application is HRSID.HRSID is a dataset for ship detection,semantic segmentation,and instance segmentation tasks in high-resolution SAR images.The dataset contains 5604 SAR images with resolutions of 0.5,1,and3m.(https://github.com/chaozhong2010/hrsid)
The results for small target detection model application are shown in the table below. The proposed hybrid asymmetric quantization method can improve the detection accuracy compared with PyTorch.

| Model | PC_Accuracy(FP32) | NNI Accuracy (fake INT8 ) | PyTorch_Accuracy_(fake_INT8) |  | Proposed_Quantization_Accuracy(fake INT 8) |  |  |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------|
|  |  |  | Moving MinMax  | MinMax | Histogram | Proposed Activation Redistribution |
| Yolo-v3 tiny | 90.70% | 85.7% | 83.2% | 85.6% | 86.7% | 86.1% | 90.12% |
## License

[Xi’an Microelectronics Technology Institute](LICENSE) © ycjcy
