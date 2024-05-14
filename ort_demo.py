import onnxruntime
import onnx
import torch
import numpy as np
from PIL import Image
import json
import cv2

def preprocess(img0_pth, img1_pth):
    img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
    #img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))  # input size shuold be divisible by 32
    img0_raw = cv2.resize(img0_raw, (640,640))  # input size shuold be divisible by 32
    #img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))
    img1_raw = cv2.resize(img1_raw, (640,640))
    img0 = img0_raw[None][None] / 255.
    img1 = img1_raw[None][None] / 255.
    return img0.astype(np.float32), img1.astype(np.float32)

if __name__ == "__main__":
    ### 读取模型,构建session
    ### 无gpu设置 providers=['CPUExecutionProvider']
    ### 有gpu设置 providers=['CUDAExecutionProvider']
    session = onnxruntime.InferenceSession('loftr_outdoor_ds1.onnx',  providers=['CUDAExecutionProvider'])

    ### 前处理
    img0_pth = r"data\1.jpg"
    img1_pth = r"data\2.jpg"
    img0_pth = "assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
    img1_pth = "assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
    img0, img1 = preprocess(img0_pth, img1_pth)

    ### 推理
    #c1, c2 = session.run(None, {'image0': img0, 'image1': img1})
    k0, k1, c = session.run(None, {'image0': img0, 'image1': img1})


    img0_raw = cv2.imread(img0_pth)
    img1_raw = cv2.imread(img1_pth)
    img0_raw = cv2.resize(img0_raw, (640,640)) 
    img1_raw =cv2.resize(img1_raw, (640,640)) 
    h0,w0,_ = img0_raw.shape   # hwc
    h1,w1,_ = img1_raw.shape

    newW = (w0+w1+50)
    newH = max(h0, h1)

    show_data = np.zeros((newH,newW,3),dtype=np.uint8)
    show_data[0:h0, 0:w0,:] = img0_raw
    show_data[0:h1, w0+50:,:] = img1_raw

    num = (c>0).sum()
    for i in range(num):
        cv2.line(show_data, (int(k0[i][0]),int(k0[i][1])), (int(k1[i][0])+w0+50,int(k1[i][1])), (0,255,0), 1)
    cv2.imwrite("x.jpg",show_data)
    # cv2.waitKey(0)