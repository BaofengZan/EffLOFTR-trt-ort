#coding=utf-8

import os
#os.chdir("..")
from copy import deepcopy

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure

import matplotlib.pyplot as plt
# outdorr
import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

import onnx
import onnx
from onnxsim import simplify



class LoFTRWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.k1 = torch.zeros((1000,2), dtype=torch.float32,requires_grad=False).cuda()
        # self.k2 = torch.zeros((1000,2), dtype=torch.float32,requires_grad=False).cuda()
        # self.c =  torch.zeros(1000, dtype=torch.float32,requires_grad=False).cuda()
    def forward(self, img1, img2):
        data = {
            "image0": img1,
            "image1": img2,
        }

        self.model(data)
        #data.update({'sim_matrix_ff': conf_matrix_ff})
        #data.update({'conf_matrix_f': softmax_matrix_f})
        # sim_matrix_ff = data['sim_matrix_ff']
        # softmax_matrix_f = data['conf_matrix_f']
        # conf_matrix = data['conf_matrix']
        mkpts0 = data['mkpts0_f']
        mkpts1 = data['mkpts1_f']
        mconf = data['mconf']
        # self.k1[:mkpts0.shape[0]] = mkpts0
        # self.k2[:mkpts1.shape[0]] = mkpts1
        # self.c[:mconf.shape[0]] = mconf
        # return self.k1,self.k2,self.c
        return mkpts0,mkpts1,mconf
        #return sim_matrix_ff,softmax_matrix_f



# You can choose model type in ['full', 'opt']
model_type = 'full' # 'full' for best quality, 'opt' for best efficiency

# You can choose numerical precision in ['fp32', 'mp', 'fp16']. 'fp16' for best efficiency
precision = 'fp32' # Enjoy near-lossless precision with Mixed Precision (MP) / FP16 computation if you have a modern GPU (recommended NVIDIA architecture >= SM_70).

# You can also change the default values like thr. and npe (based on input image size)

if model_type == 'full':
    _default_cfg = deepcopy(full_default_cfg)
elif model_type == 'opt':
    _default_cfg = deepcopy(opt_default_cfg)
    
if precision == 'mp':
    _default_cfg['mp'] = True
elif precision == 'fp16':
    _default_cfg['half'] = True
    
print(_default_cfg)
matcher = LoFTR(config=_default_cfg)

matcher.load_state_dict(torch.load(r"weights\eloftr_outdoor.ckpt")['state_dict'])
matcher = reparameter(matcher) # no reparameterization will lead to low performance

if precision == 'fp16':
    matcher = matcher.half()

matcher = matcher.eval().cuda()


onnx_export = True
if onnx_export:
    model = LoFTRWrapper(matcher).eval()
    batch_size = 1
    height = 640
    width = 640

    data = {}
    #img0_pth = "assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
    #img1_pth = "assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
    img0_pth = r"data\onnx.jpg"
    #img0_pth = "assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
    img1_pth = r"data\2.jpg"
    img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
    img0_raw = cv2.resize(img0_raw, (640, 640))  # input size shuold be divisible by 32
    img1_raw = cv2.resize(img1_raw, (640, 640))
    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    data["image0"] = img0
    data["image1"] = img1

    # 这里的图片，要选择一张要有效点尽量多的图片 
    # 所以这里输入同一张图，并且测试不同的图 选择一个最多的，1.jpg 达到5776(一共6400个)
    # united_states_capitol_26757027_6717084061.jpg 为4863个
    # onnx.jpg 5776
    # 
    torch.onnx.export(
        model,
        ( data["image0"],  data["image0"]),  # 这里弄相同的图片才能保证有足够多的点
        "loftr_outdoor_ds1.onnx",
        opset_version=17,
        input_names=list(data.keys()),
        #output_names=["sim_matrix_ff","softmax_matrix_f"],
        output_names=["keypoints0", "keypoints1", "confidence"],
        dynamic_axes={
            "image0": {2: "height", 3: "width"},
            "image1": {2: "height", 3: "width"},
            "keypoints0": {0: "num_keypoints"},
            "keypoints1": {0: "num_keypoints"},
            "confidence": {0: "num_keypoints"},
        },
        # dynamic_axes={
        #     # "image0": {2: "height", 3: "width"},
        #     # "image1": {2: "height", 3: "width"},
        #     "sim_matrix_ff": {0: "num_keypoints"},
        #     "softmax_matrix_f": {0: "num_keypoints"}
        # },
    )
    # onnx_model = onnx.load("loftr_outdoor_ds.onnx")  # load onnx model
    # model_simp, check = simplify(onnx_model)
    # assert check, "Simplified ONNX model could not be validated"
    # onnx.save(model_simp, "loftr_outdoor_ds.onnx")
    # print('finished exporting onnx')
    exit(0)

# Load example images
img0_pth = "assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
img1_pth = "assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))  # input size shuold be divisible by 32
img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))

if precision == 'fp16':
    img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.
else:
    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}

# Inference with EfficientLoFTR and get prediction
with torch.no_grad():
    if precision == 'mp':
        with torch.autocast(enabled=True, device_type='cuda'):
            matcher(batch)
    else:
        matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()
    pnumber = batch['point_num'].cpu().numpy()

mkpts0 = mkpts0[:pnumber]
mkpts1 = mkpts1[:pnumber]
mconf = mconf[:pnumber]
# Draw
if model_type == 'opt':
    print(mconf.max())
    mconf = (mconf - min(20.0, mconf.min())) / (max(30.0, mconf.max()) - min(20.0, mconf.min()))

color = cm.jet(mconf)
text = [
    'LoFTR',
    'Matches: {}'.format(len(mkpts0)),
]

fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)

plt.show()