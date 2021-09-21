import inspect
import cv2
import numpy as np
from enot_lite import backend
import onnxruntime as ort
import torch
import sys

resolution = (640, 640)
crop_start = (270, 0)
crop_size = 810

# nms parameters
confidence_threshold = 0.25
iou_threshold = 0.65

crop = (
    (crop_start[0], crop_start[0] + crop_size),
    (crop_start[1], crop_start[1] + crop_size),
)

image = cv2.imread('bus.jpg')
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image = image[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1], :3]
image = cv2.resize(image, resolution, interpolation=cv2.INTER_CUBIC)

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session_options.enable_profiling = True

sess = backend.OrtTensorrtFloatBackend('yolov5s.onnx', sess_opt=session_options)

input_name = sess.get_inputs()[0].name
net_input = np.transpose(image, (2, 0, 1))  # HWC to CHW
net_input = net_input[None, :, :, :]  # CHW to NCHW
net_input = net_input.astype(np.float32) / 255.0  # normalize to 0-1

outputs = sess.run(output_names=None, input_feed={input_name: net_input})[0]

for _ in range(120):
    outputs = sess.run(output_names=None, input_feed={input_name: net_input})[0]
    
print(sess._sess.end_profiling())
