# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:23:46 2018

@author: RAJAT
"""

import numpy as np
from time import time
from keras.models import Sequential,Model
from keras.layers import *
from keras.callbacks import TensorBoard
from model import load_model_weights
from datacollect import load_train_data , shape , margin
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model = load_model_weights("v3_nfs_ai")
layer_dict = dict([(layer.name, layer) for layer in model.layers])
X_train , Y_train = load_train_data('second_batch')

model.summary()

test = Sequential()
test.add(layer_dict['right_input'])
test.add(layer_dict['conv2d_1'])
test.add(layer_dict['conv2d_2'])
#test.add(layer_dict['conv2d_3'])
#test.add(layer_dict['conv2d_4'])
#test.add(layer_dict['conv2d_5'])
img_no = 8000
side=0
plt.imshow(X_train[side,img_no,:,:,0] , cmap='gray')

res = test.predict(X_train[side,:,:,:,:])

fig=plt.figure(figsize=(10, 10))

columns=6
rows = 4

for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(res[img_no,:,:,i]  , cmap='gray')

plt.show()
