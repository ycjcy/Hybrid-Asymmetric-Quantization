# coding=utf-8
# uncompyle6 version 3.2.4
# Python bytecode 2.7 (62211)
# Decompiled from: Python 3.6.7 |Anaconda custom (64-bit)| (default, Oct 28 2018, 19:44:12) [MSC v.1915 64 bit (AMD64)]
# Embedded file name: ./tools_binaries/quantize.py
# Compiled at: 2018-06-09 06:44:16
"""
quantize.py is an out-of-the-box network quantizer callable from the command line.
"""

from __future__ import division
import numpy as np, os, sys, argparse, math, collections, scipy, google.protobuf.text_format as tfmt, json

np.set_printoptions(threshold=1000000000)
import caffe
import copy
from google.protobuf import text_format
import re
import cv2
import torch
from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver, HistogramObserver
from quantize_pyd import execute_calibration
ans = []


##读取prototxt
def readPrototxt(Orign_Prototxt_Path):
    net = caffe.proto.caffe_pb2.NetParameter()
    text_format.Merge(open(Orign_Prototxt_Path).read(), net)
    return net


# 读取caffemodel
def readCaffemodel(Orign_Caffemodel_Path):
    model = caffe.proto.caffe_pb2.NetParameter()
    with open(Orign_Caffemodel_Path, 'rb') as f:
        model.ParseFromString(f.read())
    return model


##获取prototxt中layer
def getNetLayer(net):
    if len(net.layer):
        Layer = net.layer
    elif len(net.layers):
        Layer = net.layers
    return Layer


##获取caffemodel中layer
def getModelLayer(model):
    if len(model.layer):
        Layer = model.layer
    elif len(model.layers):
        Layer = model.layers
    return Layer


##根据index删除prototxt中layer
def delLayer(net, index):
    Layer = getNetLayer(net)
    i = 0
    for a in index:
        del Layer[a - i]
        i = i + 1


##提取Conv+BatchNorm+Scale的名字和BatchNorm、Scale在Layer列表中的index
def getConvBNLayer(net):
    Layer = getNetLayer(net)
    Conv_BN_list = []
    BN_index = []
    for i in range(len(Layer)):
        ConvBN = []
        if (i + 1) < len(Layer) and Layer[i].type == "Convolution" or Layer[i].type == 4:
            if Layer[i + 1].type == "BatchNorm" and Layer[i + 2].type == "Scale":
                ConvBN.append(Layer[i].name)
                ConvBN.append(Layer[i + 1].name)
                ConvBN.append(Layer[i + 2].name)
                Layer[i].convolution_param.bias_term = True
                BN_index.append(i + 1)
                BN_index.append(i + 2)
                Conv_BN_list.append(ConvBN)
    return Conv_BN_list, BN_index


def getParam(name, model):
    ModelLayer = getModelLayer(model)
    for i in range(len(ModelLayer)):
        if ModelLayer[i].name == name:
            global params  # wyj
            params = ModelLayer[i].blobs
            break
    return params, i


def disposeParams(Params):
    data, shape = np.array(Params.data), Params.shape.dim
    P_data = data.reshape(shape)
    return P_data


##修改参数，将Conv、BatchNorm、Scale的参数融合再存到Conv中
def modifyParam(ConvBNName, model):
    for CBN in ConvBNName:

        W_bc, conv_index = getParam(CBN[0], model)
        W_data = disposeParams(W_bc[0])

        bc_data = 0
        if len(W_bc) != 1:
            bc_data = disposeParams(W_bc[1])

        m_v, bn_index = getParam(CBN[1], model)
        m_data = disposeParams(m_v[0])
        v_data = 0
        meanFromBN = 1
        if len(m_v) != 1:
            v_data = disposeParams(m_v[1])
            # 解决BN层中mean值不为1的问题。
            if len(m_v) != 2:
                meanFromBN = disposeParams(m_v[2])

        m_data = m_data / meanFromBN
        v_data = v_data / meanFromBN
        s_bs, scale_index = getParam(CBN[2], model)
        s_data = disposeParams(s_bs[0])
        bs_data = disposeParams(s_bs[1])  # scale无bias注释掉此行

        W_new = []
        for i in range(W_data.shape[0]):
            temp = W_data[i] * s_data[i] / np.sqrt(v_data[i] + 0.00001)
            W_new.append(temp)
        b_new = (bc_data - m_data) * s_data / np.sqrt(v_data + 0.00001) + bs_data
        # b_new = (bc_data - m_data) * s_data / np.sqrt(v_data + 0.00001)    #scale无bias

        model.layer[conv_index].blobs[0].data[:] = np.array(W_new).reshape(1, -1).tolist()[0]
        if len(W_bc) == 1:  # conv无偏置bias，bc_data = 0
            model.layer[scale_index].blobs[1].data[:] = np.array(b_new).reshape(1, -1).tolist()[0]
            model.layer[conv_index].blobs.extend([model.layer[scale_index].blobs[1]])
        else:
            model.layer[conv_index].blobs[1].data[:] = np.array(b_new).reshape(1, -1).tolist()[0]

    return model
def declare_network(deploy_model, weights):
    net = caffe.Net(deploy_model, weights, caffe.TEST)
    net_parameter = caffe.proto.caffe_pb2.NetParameter()
    return (
        net, net_parameter)

def initialize_calibration(net, calibration_size, dims, calibration_filenames, calibration_indices,
                           mean_value, std_value, lstm_flag, normalized_flag, tensor_directory):
    if dims[0] == 3 or dims[0] == 1:
        net.blobs['data'].reshape(calibration_size, *dims)
        for i in range(calibration_size):
            print(calibration_filenames[calibration_indices[i]])
            if dims[0] == 3:
                # data = caffe.io.load_image(calibration_filenames[calibration_indices[i]], color=True)
                data = cv2.imread(calibration_filenames[calibration_indices[i]])
                data = cv2.resize(data, (dims[2], dims[1]))
                data = data.transpose(2, 0, 1)  # C*H*W  RGB有此行，gray注释此行
                data = data.astype(np.float32)
                ## lstm预处理先除255再减均值除方差
                if lstm_flag:
                    if normalized_flag:
                        data = data / 255.0
                    data[0, :, :] = data[0, :, :] - mean_value[0]
                    data[1, :, :] = data[1, :, :] - mean_value[1]
                    data[2, :, :] = data[2, :, :] - mean_value[2]
                    data[0, :, :] = data[0, :, :] / std_value[0]
                    data[1, :, :] = data[1, :, :] / std_value[1]
                    data[2, :, :] = data[2, :, :] / std_value[2]
                else:
                    data[0, :, :] = data[0, :, :] - mean_value[0]
                    data[1, :, :] = data[1, :, :] - mean_value[1]
                    data[2, :, :] = data[2, :, :] - mean_value[2]
                    data[0, :, :] = data[0, :, :] / std_value[0]
                    data[1, :, :] = data[1, :, :] / std_value[1]
                    data[2, :, :] = data[2, :, :] / std_value[2]
                    if normalized_flag:
                        data = data / 255.0
            else:
                # data = caffe.io.load_image(calibration_filenames[calibration_indices[i]], color=False)
                # data = transformer.preprocess('data', data)
                data = cv2.imread(calibration_filenames[calibration_indices[i]], 0)
                data = cv2.resize(data, (dims[2], dims[1]))
                data = data.reshape(1, dims[2], dims[1])
                data = data.transpose(0, 2, 1)
                data = data.astype(np.float32)
                data[:, :] = data[:, :] - mean_value
                data[:, :] = data[:, :] / std_value
                if normalized_flag:
                    data = data / 255.0

            net.blobs['data'].data[i] = data


    else:
        net.blobs['data'].reshape(10, *dims)
        data = np.loadtxt(tensor_directory, dtype=np.float32, delimiter=',').reshape(10, dims[0], dims[1],
                                                                                     dims[2])  # (N,C,H,W)
        if normalized_flag:
            data = data / 255.0
        for i in range(10):
            net.blobs['data'].data[i] = data[i]
    return net

def main(argv):
    pycaffe_dir = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy_model', default=os.path.join(pycaffe_dir,
                                                               '../model/deploy.prototxt'),
                        help='Input prototxt for calibration')
    parser.add_argument('--weights', default=os.path.join(pycaffe_dir,
                                                          '../model/deploy.caffemodel'),
                        help='FP32 pretrained caffe model')
    parser.add_argument('--fuse_flag', type=int, default=0)
    parser.add_argument('--fuse_deploy_model', default=os.path.join(pycaffe_dir,
                                                                    '../model/fuse_deploy.prototxt'),
                        help='Input fused prototxt')
    parser.add_argument('--fuse_weights', default=os.path.join(pycaffe_dir,
                                                               '../model/fuse_deploy.caffemodel'),
                        help='FP32 pretrained fused caffe model')
    parser.add_argument('--quantized_deploy_model', default=os.path.join(pycaffe_dir,
                                                                         '../quantized_deploy.prototxt'),
                        help='Output file name for calibration-deploy')
    parser.add_argument('--calibration_directory',
                        default=os.path.join(pycaffe_dir, '../images/sample_calibration_dataset/'),
                        help='Dir of dataset of original images')
    parser.add_argument('--tensor_directory',
                        default=os.path.join(pycaffe_dir, '../images/sample_calibration_dataset/'),
                        help='Dir of dataset of original tensor')
    parser.add_argument('--calibration_size', type=int, default=8,
                        help='Number of images to use for calibration, default is 8')
    parser.add_argument('--calibration_indices', default=None)
    parser.add_argument('--bitwidths', default='8,8,8', help='Bit widths for input,params,output default: 8,8,8')
    parser.add_argument('--dims', default='1,3,224,224', help='Dimensions for first layer, default 3,224,224')
    parser.add_argument('--transpose', default='2,0,1',
                        help='Passed to caffe.io.Transformer function set_transpose, default 2,0,1')
    parser.add_argument('--channel_swap', default='2,1,0',
                        help='Passed to caffe.io.Transformer function set_channel_swap, default 2,1,0')
    parser.add_argument('--raw_scale', type=float, default=255.0,
                        help='Passed to caffe.io.Transformer function set_raw_scale, default 255.0')
    parser.add_argument('--mean_value', default='0',
                        help='Passed to caffe.io.Transformer function set_mean, default 104,117,123')
    parser.add_argument('--std_value', default='1',
                        help='Passed to caffe.io.Transformer function set_mean, default 1,1,1')
    parser.add_argument('--normalized_flag', type=int, default=0)
    parser.add_argument('--relu_flag', type=int, default=1)
    parser.add_argument('--lstm_flag', type=int, default=0)
    parser.add_argument('--quantmax_name', type=str, default='',
                        help='layer name of the output feature layer quantized using the maximum value')
    parser.add_argument('--quantize_mode', type=int, default=0)
    args = parser.parse_args()
    deploy_model = None
    if args.deploy_model:
        deploy_model = args.deploy_model
    weights = None
    if args.weights:
        weights = args.weights

    fuse_flag = args.fuse_flag
    fuse_deploy_model = None
    fuse_weights = None
    if fuse_flag:
        fuse_deploy_model = args.fuse_deploy_model
        fuse_weights = args.fuse_weights

    quantized_deploy_model = None
    if args.quantized_deploy_model:
        quantized_deploy_model = args.quantized_deploy_model
    calibration_directory = None
    if args.calibration_directory:
        calibration_directory = args.calibration_directory
    tensor_directory = None
    if args.tensor_directory:
        tensor_directory = args.tensor_directory
    calibration_filenames = os.listdir(calibration_directory)
    for i in range(0, len(calibration_filenames)):
        calibration_filenames[i] = os.path.join(calibration_directory, calibration_filenames[i])
    calibration_size = None
    if args.calibration_size:
        calibration_size = int(args.calibration_size)
        if calibration_size > len(calibration_filenames):
            sys.stderr.write(
                'Requested calibration size is greater than the number of available files in the specified calibration directory\n')
            quit(1)
    calibration_indices = None
    extra_calibration_indices = None
    if args.calibration_indices:
        calibration_indices = np.array([int(s) for s in args.calibration_indices.split(',')])
        remaining_array = np.setdiff1d(np.arange(0, len(calibration_filenames)), calibration_indices)
        extra_calibration_indices = np.sort(
            np.random.choice(remaining_array, calibration_size - len(calibration_indices), replace=False))
        calibration_indices = np.sort(np.append(calibration_indices, extra_calibration_indices))
    else:
        calibration_indices = np.sort(
            np.random.choice(range(len(calibration_filenames)), calibration_size, replace=False))
    bitwidths = None
    if args.bitwidths:
        bitwidths = [int(s) for s in args.bitwidths.split(',')]
    dims = None
    if args.dims:
        dims = [int(s) for s in args.dims.split(',')]
    transpose = None
    if args.transpose:
        transpose = [int(s) for s in args.transpose.split(',')]
    channel_swap = None
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]
    raw_scale = None
    if args.raw_scale:
        raw_scale = float(args.raw_scale)
    mean_value = None
    if args.mean_value:
        mean_value_list = [float(s) for s in args.mean_value.split(',')]
        mean_value = np.array(mean_value_list)
    std_value = None
    if args.std_value:
        std_value_list = [float(s) for s in args.std_value.split(',')]
        std_value = np.array(std_value_list)
    normalized_flag = None
    if args.normalized_flag:
        normalized_flag = args.normalized_flag
    relu_flag = None
    if args.relu_flag:
        relu_flag = args.relu_flag
    lstm_flag = None
    if args.lstm_flag:
        lstm_flag = float(args.lstm_flag)

    quantmax_name = None
    if args.quantmax_name:
        quantmax_name = [str(s) for s in args.quantmax_name.split(',')]
        # quantmax_name = np.array(quantmax_name_list)
    quantize_mode = [MinMaxObserver, MovingAverageMinMaxObserver, HistogramObserver]
    quantize_idx = args.quantize_mode
    caffe.set_mode_cpu()

    if fuse_flag:
        ############fuse conv+bn+scale###########
        # readPrototxt
        deploy = readPrototxt(deploy_model)
        # readCaffemodel
        model = readCaffemodel(weights)
        ## 提取Conv+BatchNorm+Scale的名字和BatchNorm、Scale在Layer列表中的index
        Conv_Bn_Scale_Name, CBSN_index = getConvBNLayer(deploy)
        delLayer(deploy, CBSN_index)
        # 保存修改后的prototxt
        # with open(fuse_deploy_model, 'w') as (f):
        #     f.write(tfmt.MessageToString(deploy))
        open(fuse_deploy_model, 'w').write(tfmt.MessageToString(deploy))

        ##修改参数，将Conv、BatchNorm、Scale的参数融合再存到Conv中
        model = modifyParam(Conv_Bn_Scale_Name, model)
        # 保存修改后的caffemodel
        # with open(fuse_weights, "wb") as f:
        #     f.write(model.SerializeToString())
        open(fuse_weights, "wb").write(model.SerializeToString())
        deploy_model = fuse_deploy_model
        weights = fuse_weights
        ############fuse conv+bn+scale###########

    net, net_parameter = declare_network(deploy_model, weights)
    net = initialize_calibration(net, calibration_size, dims, calibration_filenames, calibration_indices,
                                 mean_value, std_value, lstm_flag, normalized_flag, tensor_directory)
    execute_calibration(net, net_parameter, bitwidths, deploy_model, relu_flag, quantized_deploy_model,
                        quantmax_name,quantize_mode[quantize_idx])
    return


if __name__ == '__main__':
    import time

    start = time.clock()
    main(sys.argv)
    end = (time.clock() - start)
    print("time used:", end)
