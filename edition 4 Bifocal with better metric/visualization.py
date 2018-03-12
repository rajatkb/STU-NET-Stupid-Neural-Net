# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:23:46 2018

@author: RAJAT
"""

import numpy as np
import cv2
from time import time
from keras.models import Sequential,Model
from keras.layers import *
from keras.callbacks import TensorBoard
from model import load_model_weights
from datacollect import load_train_data , shape , margin
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model = load_model_weights("v4_nfs_iteration2_ai")
layer_dict = dict([(layer.name, layer) for layer in model.layers])
X_train , Y_train = load_train_data('third_batch')

model.summary()

test = Sequential()
test.add(layer_dict['right_input'])
test.add(layer_dict['conv2d_1'])
test.add(layer_dict['conv2d_2'])
#test.add(layer_dict['conv2d_3'])
#test.add(layer_dict['conv2d_4'])
#test.add(layer_dict['conv2d_5'])
side=1

res = test.predict(X_train[side,:,:,:,:])

for i in range(X_train.shape[1]):
    frame = np.reshape(X_train[side,i,:,:], (shape[0],shape[1]-margin))
    cv2.imshow('frame' , frame)
    for i in range(X_train.shape[-1]):
    	frame = np.reshape(X_train[side,i,:,:,:], (X_train[side][i].shape[0],X_train[side][i].shape[1]))
    	cv2.imshow('predict '+str(i) , frame)
    cv2.waitKey(1)

cv2.destroyAllWindows()
del(X_train)
del(Y_train)
