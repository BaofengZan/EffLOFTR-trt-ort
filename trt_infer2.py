import ctypes

import numpy as np
import tensorrt as trt
from cuda import cudart
import cv2

trtFile = "./demo.plan"
timeCacheFile = "./model.cache"
nB, nC, nH, nW = 1, 1, 640, 640
np.random.seed(31193)

def preprocess(img0_raw, img1_raw):
    #img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))  # input size shuold be divisible by 32
    #img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))
    img0_raw = cv2.resize(img0_raw, (640, 640))  # input size shuold be divisible by 32
    img1_raw = cv2.resize(img1_raw,  (640, 640))
    img0 = img0_raw[None][None] / 255.
    img1 = img1_raw[None][None] / 255.
    return img0.astype(np.float32), img1.astype(np.float32)


img0_pth = "assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
img1_pth = "assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
data1, data2 = preprocess(img0_raw, img1_raw)

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()

trt.init_libnvinfer_plugins(logger, namespace="")
with open(trtFile, 'rb') as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
#engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context_without_device_memory()  # do not alloc GPU memory when creating the context
print("Device memory needed by engine is %d byte" % engine.device_memory_size)
status, address = cudart.cudaMalloc(engine.device_memory_size)  # alloc GPU memory by ourselves
context.device_memory = address  # assign the address to the context

context.set_input_shape(lTensorName[0], [nB, nC, nH, nW])
context.set_input_shape(lTensorName[1], [nB, nC, nH, nW])
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

bufferH = []
bufferH.append(np.ascontiguousarray(data1))
bufferH.append(np.ascontiguousarray(data2))
#for i in range(nInput, nIO):
#   bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))

bufferD = []
for i in range(nInput):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

#for i in range(nInput, nIO):
bufferD.append(cudart.cudaMalloc(4*6400)[1])
bufferH.append(np.empty((6400,), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[0]))))
bufferD.append(cudart.cudaMalloc(4*6400*2)[1])
bufferH.append(np.empty((6400,2), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[0]))))
bufferD.append(cudart.cudaMalloc(4*6400*2)[1])
bufferH.append(np.empty((6400,2), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[0]))))

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

for i in range(nIO):
    context.set_tensor_address(lTensorName[i], int(bufferD[i]))

context.execute_async_v3(0)

for i in range(nInput, nIO):
    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

for i in range(nIO):
    print(lTensorName[i])
    print(bufferH[i])


# 画图
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

k0 = bufferH[3]
k1 = bufferH[4]
num = (bufferH[2]>0).sum()
for i in range(num):
    cv2.line(show_data, (int(k0[i][0]),int(k0[i][1])), (int(k1[i][0])+w0+50,int(k1[i][1])), (0,255,0), 1)
cv2.imwrite("x.jpg",show_data)

for b in bufferD:
    cudart.cudaFree(b)