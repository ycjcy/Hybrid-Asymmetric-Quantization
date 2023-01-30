# coding=utf-8
# USAGE
# python pi_deep_learning.py --prototxt models/bvlc_googlenet.prototxt --model models/bvlc_googlenet.caffemodel --labels synset_words.txt --image images/barbershop.png
# python pi_deep_learning.py --prototxt models/squeezenet_v1.0.prototxt --model models/squeezenet_v1.0.caffemodel --labels synset_words.txt --image images/barbershop.png

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import glob
import sys

import caffe

# caffe.set_mode_gpu()
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, default='model/image_1w',
                help="path to input image")
ap.add_argument("-j", "--result", required=False, default='model/pc_result',
                help="path to board result")
ap.add_argument("-p", "--prototxt", required=False,
                default='model/vgg16/VGG16-net.prototxt',
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=False,
                default='model/vgg16/VGG16-net-0414.caffemodel',
                help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=False, default='synset_words.txt',
                help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())
# load the class labels from disk
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# load the input image from disk
path = args["image"]
net = caffe.Net(args["prototxt"], args["model"], caffe.TEST)  # 加载model和network
print("[INFO] loading model...")

for jpg_file in glob.glob("{}/*".format(path)):
    image = cv2.imread(jpg_file)
    blob = cv2.resize(image, (224, 224)).astype(np.float32)
    blob = blob.transpose(2, 0, 1)
    mean = [104, 117, 123]  ###mobilenet 减均值后乘0.017
    for ch in range(blob.shape[0]):
        blob[ch] = (blob[ch] - mean[ch])
    # our CNN requires fixed spatial dimensions for our input image(s)
    # so we need to ensure it is resized to 224x224 pixels while
    # performing mean subtraction (104, 117, 123) to normalize the input;
    # after executing this command our "blob" now has the shape:
    # (1, 3, 224, 224)

    ##blob = cv2.dnn.blobFromImage(image, 1, (256, 256), (104, 117, 123))

    # load our serialized model from disk
    # net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


    net.blobs['data'].data[...] = blob  # 执行上面设置的图片预处理操作，并将图片载入到blob中
    # net.forward()

    # set the blob as input to the network and perform a forward-pass to
    # obtain our output classification

    start = time.time()
    preds = net.forward(end = "prob")
    end = time.time()
    print("[INFO] classification took {:.5} seconds".format(end - start))

    # sort the indexes of the probabilities in descending order (higher
    # probabilitiy first) and grab the top-5 predictions

    preds = preds["prob"].reshape((1, len(classes)))
    jpg_name = jpg_file.split("\\")[1].split(".")[0]
    print(jpg_name)
    np.savetxt("model/pc_result/"+jpg_name +".txt",preds,fmt = "%f",delimiter= " ")
    ####preds = preds.reshape((1, 1024))
    # idxs = np.argsort(preds[0])[::-1][:5]

    # loop over the top-5 predictions and display them
    # for (i, idx) in enumerate(idxs):
    #     # draw the top prediction on the input image
    #     if i == 0:
    #         text = "Label: {}, {:.2f}%".format(classes[idx],
    #                                            preds[0][idx] * 100)
    #         cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #
    #     # display the predicted label + associated probability to the
    #     # console
    #     print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
    #                                                             classes[idx], preds[0][idx]))
    #
    # # display the output image
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
