import cv2
import numpy as np

import utils.utils as utils
from net.inception import InceptionResNetV1
from net.mtcnn import mtcnn

if __name__ == "__main__":
    #------------------------------#
    #   门限函数
    #------------------------------#
    threshold = [0.5,0.7,0.8]
    #------------------------------#
    #   创建mtcnn对象
    #------------------------------#
    mtcnn_model = mtcnn()

    #------------------------------#
    #   读取图片并检测人脸
    #------------------------------#
    img = cv2.imread('face_dataset/timg.jpg')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    rectangles = mtcnn_model.detectFace(img, threshold)

    draw = img.copy()
    #------------------------------#
    #   转化成正方形
    #------------------------------#
    rectangles = utils.rect2square(np.array(rectangles))

    #------------------------------#
    #   载入facenet
    #------------------------------#
    model_path = './model_data/facenet_keras.h5'
    facenet_model = InceptionResNetV1()
    facenet_model.load_weights(model_path)

    for rectangle in rectangles:
        #---------------#
        #   截取图像
        #---------------#
        landmark = np.reshape(rectangle[5:15], (5,2)) - np.array([int(rectangle[0]), int(rectangle[1])])
        crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        #-----------------------------------------------#
        #   利用人脸关键点进行人脸对齐
        #-----------------------------------------------#
        cv2.imshow("before", cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
        crop_img,_ = utils.Alignment_1(crop_img,landmark)
        cv2.imshow("two eyes", cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

        crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)
        feature1 = utils.calc_128_vec(facenet_model,crop_img)
        print(feature1)

    cv2.waitKey(0)
