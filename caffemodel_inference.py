
import caffe

print(caffe.__path__)
import cv2
import numpy as np

model_old = "model/yolov3-tiny-1019/yolov3-tiny-1019.caffemodel"
modelconfig_old = "model/yolov3-tiny-1019/yolov3-tiny-1019.prototxt"

net = caffe.Net(modelconfig_old, caffe.TEST)
net.save("model/yolov3-tiny-1019/yolov3-tiny-1019.caffemodel")

# model_old = "model_prune_2_name_1124.caffemodel"
# modelconfig_old = "model_prune_2_name_1124.prototxt"
#
# net = caffe.Net(modelconfig_old, model_old, caffe.TEST)

image = cv2.imread("model/image/ILSVRC2012_val_00000043.JPEG")
img = cv2.resize(image, (224,224))
# image_input = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT,value = 0)
image1 = img.transpose(2, 0, 1).astype(np.float32)
# image1 = image1[0].copy()
# image1 = image1 - 127
# image1 = image1 /255.0
# image1 = image1.reshape(1, image1.shape[0], image1.shape[1])

# np.savetxt("input.txt", image1.reshape(-1, 1), fmt="%.6f")
image1[0] = image1[0]-104
image1[1] = image1[1] -117
image1[2] = image1[2] -123
blob = image1

net.blobs["data"].data[...] = blob
print("forward:")
conv1 = net.forward(end="fc8")
# conv2 = net.forward(end="layer20-upsample1")
np.savetxt("caffe_output.txt", conv1['fc8'].reshape(-1, 1), fmt="%.6f")
