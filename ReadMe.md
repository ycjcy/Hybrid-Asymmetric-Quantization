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



div.Section0{page:Section0;}</style></head><body style="tab-interval:21pt;text-justify-trim:punctuation;" ><!--StartFragment--><div class="Section0"  style="layout-grid:15.6000pt;" ><div align=center ><table class=MsoNormalTable  border=1  cellspacing=0  style="border-collapse:collapse;width:382.6000pt;mso-table-layout-alt:fixed;
border:none;mso-border-left-alt:0.2500pt solid windowtext;mso-border-top-alt:0.2500pt solid windowtext;
mso-border-right-alt:0.2500pt solid windowtext;mso-border-bottom-alt:0.2500pt solid windowtext;mso-border-insideh:0.2500pt solid windowtext;
mso-border-insidev:0.2500pt solid windowtext;mso-padding-alt:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;" ><tr style="height:5.9000pt;" ><td width=75  valign=center  rowspan=2  style="width:56.5500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=17 ><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:8.0000pt;" >Model</span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:8.0000pt;" ></span></b></p></td><td width=66  valign=center  rowspan=2  style="width:49.6500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=17 ><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:8.0000pt;" >PC </span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:8.0000pt;" >Accuracy</span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:8.0000pt;" >(FP32)</span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:8.0000pt;" ></span></b></p></td><td width=94  valign=center  rowspan=2  style="width:70.8500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=17 ><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:8.0000pt;" >PyTorch Accuracy </span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:8.0000pt;" ></span></b></p><p class=17 ><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:8.0000pt;" >(fake INT 8)</span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:8.0000pt;" ></span></b></p></td><td width=274  valign=center  colspan=4  style="width:205.5500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=17 ><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:8.0000pt;" >Proposed Hybrid Asymmetric Quantization Accuracy</span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:8.0000pt;" ></span></b></p><p class=17 ><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:8.0000pt;" >(fake INT 8)</span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:8.0000pt;" ></span></b></p></td></tr><tr style="height:5.9000pt;" ><td width=66  valign=center  style="width:49.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=15 ><b><i><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-weight:bold;font-style:italic;
font-size:7.5000pt;" ><font face="Times New Roman" >MinMax</font></span></i></b><b><i><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-style:italic;font-size:7.5000pt;" ></span></i></b></p></td><td width=56  valign=center  style="width:42.5500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=15 ><b><i><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-weight:bold;font-style:italic;
font-size:7.5000pt;" ><font face="Times New Roman" >Moving</font></span></i></b><b><i><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-style:italic;font-size:7.5000pt;" ></span></i></b></p><p class=15 ><b><i><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-weight:bold;font-style:italic;
font-size:7.5000pt;" ><font face="Times New Roman" >MinMax</font></span></i></b><b><i><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-style:italic;font-size:7.5000pt;" ></span></i></b></p></td><td width=66  valign=center  style="width:49.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=15 ><b><i><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-weight:bold;font-style:italic;
font-size:7.5000pt;" ><font face="Times New Roman" >Histogram</font></span></i></b><b><i><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-style:italic;font-size:7.5000pt;" ></span></i></b></p></td><td width=85  valign=center  style="width:63.8000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:1.0000pt solid windowtext;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=15 ><b><i><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-style:italic;font-size:7.5000pt;" >P</span></i></b><b><i><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-weight:bold;font-style:italic;
font-size:7.5000pt;" ><font face="Times New Roman" >roposed</font></span></i></b><b><i><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-style:italic;font-size:7.5000pt;" ><span style="mso-spacerun:'yes';" >&nbsp;</span>Activation Redistribution</span></i></b><b><i><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-style:italic;font-size:7.5000pt;" ></span></i></b></p></td></tr><tr style="height:7.9500pt;" ><td width=75  valign=center  style="width:56.5500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=16  align=center  style="text-align:center;" ><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" >GoogleNet</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:4.0000pt;" ></span></p></td><td width=66  valign=center  style="width:49.6500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=16  align=center  style="text-align:center;" ><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:8.0000pt;" ><font face="Times New Roman" >6</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" >7.04%</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" ></span></p></td><td width=94  valign=center  style="width:70.8500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=16  align=center  style="text-align:center;" ><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:8.0000pt;" ><font face="Times New Roman" >66.50%</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" ></span></p></td><td width=66  valign=center  style="width:49.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=16  align=center  style="text-align:center;" ><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:8.0000pt;" ><font face="Times New Roman" >66.82%</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" ></span></p></td><td width=56  valign=center  style="width:42.5500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=16  align=center  style="text-align:center;" ><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:8.0000pt;" ><font face="Times New Roman" >66.82%</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" ></span></p></td><td width=66  valign=center  style="width:49.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=MsoNormal ><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:8.0000pt;" ><font face="Times New Roman" >66.77%</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" ></span></p></td><td width=85  valign=center  style="width:63.8000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=MsoNormal ><b><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-weight:bold;font-size:8.0000pt;" ><font face="Times New Roman" >66.85%</font></span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:8.0000pt;" ></span></b></p></td></tr><tr style="height:7.9500pt;" ><td width=75  valign=center  style="width:56.5500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=16  align=center  style="text-align:center;" ><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" >MobileNetV2</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" ></span></p></td><td width=66  valign=center  style="width:49.6500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=16  align=center  style="text-align:center;" ><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:8.0000pt;" ><font face="Times New Roman" >7</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" >0.24%</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" ></span></p></td><td width=94  valign=center  style="width:70.8500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=16  align=center  style="text-align:center;" ><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:8.0000pt;" ><font face="Times New Roman" >66.82%</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" ></span></p></td><td width=66  valign=center  style="width:49.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=16  align=center  style="text-align:center;" ><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:8.0000pt;" ><font face="Times New Roman" >66.</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" >23</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:8.0000pt;" ><font face="Times New Roman" >%</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" ></span></p></td><td width=56  valign=center  style="width:42.5500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=16  align=center  style="text-align:center;" ><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:8.0000pt;" ><font face="Times New Roman" >66.</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" >60</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:8.0000pt;" ><font face="Times New Roman" >%</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" ></span></p></td><td width=66  valign=center  style="width:49.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=MsoNormal ><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:8.0000pt;" ><font face="Times New Roman" >66.99%</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" ></span></p></td><td width=85  valign=center  style="width:63.8000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=MsoNormal ><b><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-weight:bold;font-size:8.0000pt;" ><font face="Times New Roman" >67.38%</font></span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:8.0000pt;" ></span></b></p></td></tr><tr style="height:7.9500pt;" ><td width=75  valign=center  style="width:56.5500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=16  align=center  style="text-align:center;" ><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" >VGG16</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" ></span></p></td><td width=66  valign=center  style="width:49.6500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=16  align=center  style="text-align:center;" ><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:8.0000pt;" ><font face="Times New Roman" >6</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" >6.13%</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" ></span></p></td><td width=94  valign=center  style="width:70.8500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=16  align=center  style="text-align:center;" ><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:8.0000pt;" ><font face="Times New Roman" >65.22%</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" ></span></p></td><td width=66  valign=center  style="width:49.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=16  align=center  style="text-align:center;" ><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:8.0000pt;" ><font face="Times New Roman" >65.12%</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" ></span></p></td><td width=56  valign=center  style="width:42.5500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=16  align=center  style="text-align:center;" ><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:8.0000pt;" ><font face="Times New Roman" >66.20%</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" ></span></p></td><td width=66  valign=center  style="width:49.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=MsoNormal ><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:8.0000pt;" ><font face="Times New Roman" >65.31%</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:8.0000pt;" ></span></p></td><td width=85  valign=center  style="width:63.8000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
mso-border-left-alt:0.2500pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.2500pt solid windowtext;
border-top:none;mso-border-top-alt:0.2500pt solid windowtext;border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.2500pt solid windowtext;" ><p class=MsoNormal ><b><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-weight:bold;font-size:8.0000pt;" ><font face="Times New Roman" >66.26%</font></span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:8.0000pt;" ></span></b></p></td></tr></table></div><p class=MsoNormal ><span style="mso-spacerun:'yes';font-family:'Times New Roman';mso-fareast-font-family:宋体;" >&nbsp;</span></p></div><!--EndFragment--></body></html>
## License

[Xi’an Microelectronics Technology Institute](LICENSE) © ycjcy
