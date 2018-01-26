# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 02:22:23 2018

@author: RAJAT
"""

import numpy as np
from time import time
from keras.models import Sequential
from keras.layers import Dense, Dropout , Activation,Flatten,Convolution2D, MaxPooling2D ,AveragePooling2D
from keras.callbacks import TensorBoard
# This is where we create the model for training purpose
# 1.  we create a function self_driving
#     returns a model
# 2.  we create a function to save
move_encode = {
            'W':np.array([1,0,0,0,0]).reshape(1,5),
            'S':np.array([0,1,0,0,0]).reshape(1,5),
            'A':np.array([0,0,1,0,0]).reshape(1,5),
            'D':np.array([0,0,0,1,0]).reshape(1,5),
            '.':np.array([0,0,0,0,1]).reshape(1,5),
        }

move_decode ={
            0:'W',        
            1:'S',
            2:'A',
            3:'D',
            4:'.'
        }


def encode_movement(move):
    try:
        return move_encode[move[0]]
    except (KeyError , IndexError) as e:
        return move_encode['.']
    

def self_driving(shape):
    model = Sequential()
    model.add(Convolution2D(512 , kernel_size=(2,2), strides=(1,1) , padding='valid' , activation='relu', input_shape=shape))
    model.add(Convolution2D(128 , kernel_size=(1,1), strides=(1,1) , padding='valid' , activation='relu', input_shape=shape))
    model.add(Convolution2D(256 , kernel_size=(2,2), strides=(1,1) , padding='valid' , activation='relu', input_shape=shape))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Convolution2D(64 , kernel_size=(1,1), strides=(1,1) , padding='valid' , activation='relu', input_shape=shape))
    model.add(Convolution2D(128 , kernel_size=(2,2), strides=(2,2) , padding='valid' , activation='relu', input_shape=shape))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Convolution2D(32 , kernel_size=(1,1), strides=(1,1) , padding='valid' , activation='relu', input_shape=shape))
    model.add(Convolution2D(64 , kernel_size=(2,2), strides=(3,3) , padding='valid' , activation='relu', input_shape=shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def save_model(model,name):
     model_json = model.to_json()
     with open(name+".json", "w") as json_file:
         json_file.write(model_json)
     model.save_weights(name+".h5")
     print("\n SAVED THE WEIGHTS AND MODEL !!!!")

def load_model_weights(name):
    from keras.models import model_from_json
    json_file = open(name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(name+".h5")
    loaded_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return loaded_model

if __name__ == '__main__':
    from datacollect import load_train_data
    X_train , Y_train = load_train_data()
    model = load_model_weights('v1_nfs_ai')
    #model = self_driving(X_train[0].shape)
    for i in range(100):
        model.fit(X_train, Y_train, 
              batch_size=30,nb_epoch=1, verbose=1)
        save_model(model , "v1_nfs_ai")


    