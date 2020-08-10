from __future__ import print_function

import matplotlib.pyplot as plot
import importlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opencv_transforms import transforms
from torch.autograd import Variable

from PIL import Image

from option import Options
from utils import *
import sys
import os
import cv2
import time

PATH = "./output/"

# init the args
global best_pred, errlist_train, errlist_val
cuda = True
torch.manual_seed(1)

# init the model
models = importlib.import_module('model.'+ 'DEPnet')
model = models.Net(23)
model.cuda()
# Please use CUDA_VISIBLE_DEVICES to control the number of gpus
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(PATH + 'test_model.pth'))
#print(model)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

camera = cv2.VideoCapture(0)
time.sleep(1)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prevTime = 0

classes = ['brick', 'carpet', 'ceramic', 'fabric', 'foliage', 'food', 'glass', 'hair',
           'leather', 'metal', 'mirror', 'other', 'painted', 'paper', 'plastic', 'polishedstone',
           'skin', 'sky', 'stone', 'tile', 'wallpaper', 'water', 'wood']

while True:
    # Load frame from the camera
    ret, frame = camera.read()
    img = transform(frame)
    img = img.unsqueeze(0)

    output = model(img)
    pred = torch.argmax(output, 1)
    prediction = ('The input picture is classified as [%s], with probability %.3f.'%
     (classes[pred.tolist()[0]], (torch.nn.functional.softmax(output, dim = 1)[0][pred.tolist()[0]]).tolist()))

	# Calculate FPS
    curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1/(sec)
    str = "FPS : %0.1f" % fps
    cv2.putText(frame, str, (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    cv2.putText(frame, prediction, (100, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    cv2.imshow("VideoFrame", frame)
    if cv2.waitKey(1)>0: break

camera.release()
cv2.destroyAllWindows()
sys.exit()
