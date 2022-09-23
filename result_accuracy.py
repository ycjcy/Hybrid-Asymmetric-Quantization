# coding=utf-8
# USAGE
# python pi_deep_learning.py --prototxt models/bvlc_googlenet.prototxt --model models/bvlc_googlenet.caffemodel --labels synset_words.txt --image images/barbershop.png
# python pi_deep_learning.py --prototxt models/squeezenet_v1.0.prototxt --model models/squeezenet_v1.0.caffemodel --labels synset_words.txt --image images/barbershop.png

# import the necessary packages
import numpy as np
import os
import argparse
import time
import cv2
import glob
import sys

import caffe

caffe.set_mode_gpu()
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--path", required=False, default='images',
                help="path to input image")
ap.add_argument("-j", "--result", required=False, default='model/vgg16_result_minmax_pytorch_0922',
                help="path to model result")
ap.add_argument("-label", "--labels", required=False, default="./val.txt",
                help="path to model label")
ap.add_argument("-l", "--classes", required=False, default='./synset_words.txt',
                help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())

# load the class labels from disk
rows = open(args["classes"]).read().strip().split("\n")
# classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
classes = [r.split(" ")[0] for r in rows]

# load our serialized model from disk
print("[INFO] loading model...")
# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# net = caffe.Net(args["prototxt"], args["model"], caffe.TEST)  # 加载model和network

# path = args["path"]
# for jpg_file in glob.glob("{}/*jpg".format(path)):

path = args["result"]
txt_root = args["labels"]
val_label = np.loadtxt(txt_root, dtype=np.str)
# txt_root_paths = os.listdir(txt_root)
result_root = os.listdir(path)
total_result = len(result_root)
true_flag = 0
num = 0
for jpg_file in result_root:
    jpg_file_path = os.path.join(path, jpg_file)
    preds = np.loadtxt(jpg_file_path).reshape((1, 1000))
    # if jpg_file =='ILSVRC2012_val_00000076':
    #     print('11111111')
    idxs = np.argmax(preds[0])
    jpg_file_name = jpg_file.split('.')[0]
    label = int(val_label[num][1])

    if idxs == label:
        true_flag += 1
        print(jpg_file)
        # print("true")
    num += 1

precision = true_flag / total_result
print('************************************************')
print('precision:', precision)
# for (i, idx) in enumerate(idxs):
#     # draw the top prediction on the input image
#     if i == 0:
#         text = "Label: {}, {:.2f}%".format(classes[idx],
#                                            preds[0][idx] * 100)
#         # cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
#         #             0.7, (0, 0, 255), 2)
#
#     # display the predicted label + associated probability to the
#     # console
#     print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
#                                                             classes[idx], preds[0][idx]))

# display the output image
# cv2.imshow("Image", image)
# cv2.waitKey(0)
