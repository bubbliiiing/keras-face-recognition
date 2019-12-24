import cv2
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
from net.inception import InceptionResNetV1


img = cv2.imread('face_dataset/timg.jpg')

# 创建mtcnn对象
mtcnn_model = mtcnn()
# 门限函数
threshold = [0.5,0.7,0.9]
# 检测人脸
rectangles = mtcnn_model.detectFace(img, threshold)

draw = img.copy()
# 转化成正方形
rectangles = utils.rect2square(np.array(rectangles))

# 载入facenet
facenet_model = InceptionResNetV1()
# model.summary()
model_path = './model_data/facenet_keras.h5'
facenet_model.load_weights(model_path)


for rectangle in rectangles:
    if rectangle is not None:
        landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160

        crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        
        crop_img = cv2.resize(crop_img,(160,160))
        cv2.imshow("before",crop_img)
        new_img,_ = utils.Alignment_1(crop_img,landmark)
        cv2.imshow("two eyes",new_img)

        # std_landmark = np.array([[54.80897114,59.00365493],
        #                         [112.01078961,55.16622207],
        #                         [86.90572522,91.41657571],
        #                         [55.78746897,114.90062758],
        #                         [113.15320624,111.08135986]])
        # crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        # crop_img = cv2.resize(crop_img,(160,160))
        # new_img,_ = utils.Alignment_2(crop_img,std_landmark,landmark)
        # cv2.imshow("affine",new_img)    
        new_img = np.expand_dims(new_img,0)
        feature1 = utils.calc_128_vec(facenet_model,new_img)

cv2.waitKey(0)