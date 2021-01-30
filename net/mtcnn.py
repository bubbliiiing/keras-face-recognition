import cv2
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Conv2D, Dense, Flatten, Input, MaxPool2D,
                          Permute, Reshape)
from keras.layers.advanced_activations import PReLU
from keras.models import Model, Sequential

from utils import utils


#-----------------------------#
#   粗略获取人脸框
#   输出bbox位置和是否有人脸
#-----------------------------#
def create_Pnet(weight_path):
    inputs = Input(shape=[None, None, 3])

    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1,2],name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU2')(x)

    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU3')(x)

    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    # 无激活函数，线性。
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    model = Model([inputs], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

#-----------------------------#
#   mtcnn的第二段
#   精修框
#-----------------------------#
def create_Rnet(weight_path):
    inputs = Input(shape=[24, 24, 3])
    # 24,24,3 -> 22,22,28 -> 11,11,28
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)

    # 11,11,28 -> 9,9,48 -> 4,4,48
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 4,4,48 -> 3,3,64
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)

    # 3,3,64 -> 64,3,3
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)

    # 576 -> 128
    x = Dense(128, name='conv4')(x)
    x = PReLU( name='prelu4')(x)

    # 128 -> 2
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    # 128 -> 4
    bbox_regress = Dense(4, name='conv5-2')(x)

    model = Model([inputs], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

#-----------------------------#
#   mtcnn的第三段
#   精修框并获得五个点
#-----------------------------#
def create_Onet(weight_path):
    inputs = Input(shape = [48,48,3])
    # 48,48,3 -> 46,46,32 -> 23,23,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1,2],name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # 23,23,32 -> 21,21,64 -> 10,10,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 8,8,64 -> 4,4,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)

    # 4,4,64 -> 3,3,128
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1,2],name='prelu4')(x)

    # 3,3,128 -> 128,12,12
    x = Permute((3,2,1))(x)
    x = Flatten()(x)

    # 1152 -> 256
    x = Dense(256, name='conv5') (x)
    x = PReLU(name='prelu5')(x)

    # 256 -> 2 
    classifier = Dense(2, activation='softmax',name='conv6-1')(x)
    # 256 -> 4 
    bbox_regress = Dense(4,name='conv6-2')(x)
    # 256 -> 10 
    landmark_regress = Dense(10,name='conv6-3')(x)

    model = Model([inputs], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)
    return model

class mtcnn():
    def __init__(self):
        self.Pnet = create_Pnet('model_data/pnet.h5')
        self.Rnet = create_Rnet('model_data/rnet.h5')
        self.Onet = create_Onet('model_data/onet.h5')

    def detectFace(self, img, threshold):
        #-----------------------------#
        #   归一化
        #-----------------------------#
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape
        #-----------------------------#
        #   计算原始输入图像
        #   每一次缩放的比例
        #-----------------------------#
        scales = utils.calculateScales(img)

        out = []
        #-----------------------------#
        #   粗略计算人脸框
        #   pnet部分
        #-----------------------------#
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(copy_img, (ws, hs))
            inputs = np.expand_dims(scale_img, 0)
            ouput = self.Pnet.predict(inputs)
            #---------------------------------------------#
            #   每次选取图像金字塔中的一张图片进行预测
            #   预测结果也是一张图片的，
            #   所以我们可以将对应的batch_size维度给消除掉
            #---------------------------------------------#
            ouput = [ouput[0][0], ouput[1][0]]
            out.append(ouput)

        rectangles = []
        #-------------------------------------------------#
        #   在这个地方我们对图像金字塔的预测结果进行循环
        #   取出每张图片的种类预测和回归预测结果
        #-------------------------------------------------#
        for i in range(len(scales)):
            #------------------------------------------------------------------#
            #   为了方便理解，这里和视频上看到的不太一样
            #   因为我们在上面对图像金字塔循环的时候就把batch_size维度给去掉了
            #------------------------------------------------------------------#
            cls_prob = out[i][0][:, :, 1]
            roi = out[i][1]
            #-------------------------------------#
            #   取出每个缩放后图片的高宽
            #-------------------------------------#
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            #-------------------------------------#
            #   解码的过程
            #-------------------------------------#
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle)

        #-------------------------------------#
        #   进行非极大抑制
        #-------------------------------------#
        rectangles = np.array(utils.NMS(rectangles, 0.7))

        if len(rectangles) == 0:
            return rectangles

        #-----------------------------#
        #   稍微精确计算人脸框
        #   Rnet部分
        #-----------------------------#
        predict_24_batch = []
        for rectangle in rectangles:
            #------------------------------------------#
            #   利用获取到的粗略坐标，在原图上进行截取
            #------------------------------------------#
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            #-----------------------------------------------#
            #   将截取到的图片进行resize，调整成24x24的大小
            #-----------------------------------------------#
            scale_img = cv2.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)

        cls_prob, roi_prob = self.Rnet.predict(np.array(predict_24_batch))
        #-------------------------------------#
        #   解码的过程
        #-------------------------------------#
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])

        if len(rectangles) == 0:
            return rectangles

        #-----------------------------#
        #   计算人脸框
        #   onet部分
        #-----------------------------#
        predict_batch = []
        for rectangle in rectangles:
            #------------------------------------------#
            #   利用获取到的粗略坐标，在原图上进行截取
            #------------------------------------------#
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            #-----------------------------------------------#
            #   将截取到的图片进行resize，调整成48x48的大小
            #-----------------------------------------------#
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)

        cls_prob, roi_prob, pts_prob = self.Onet.predict(np.array(predict_batch))
        
        #-------------------------------------#
        #   解码的过程
        #-------------------------------------#
        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])

        return rectangles

