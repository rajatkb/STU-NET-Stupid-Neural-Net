# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 02:22:23 2018

@author: RAJAT
"""

import numpy as np
from time import time
from keras.models import Sequential,Model
from keras.layers import Input ,Dense, Dropout , Activation,Flatten,Convolution2D, MaxPooling2D ,AveragePooling2D,concatenate
from keras.callbacks import TensorBoard
# This is where we create the model for training purpose
# 1.  we create a function self_driving
#     returns a model
# 2.  we create a function to save
move_encode = {
            'W' :np.array([1,0,0,0,0,0,0,0,0]).reshape(1,9),
            'AW':np.array([0,1,0,0,0,0,0,0,0]).reshape(1,9),
            'DW':np.array([0,0,1,0,0,0,0,0,0]).reshape(1,9),
            'S' :np.array([0,0,0,1,0,0,0,0,0]).reshape(1,9),
            'AS' :np.array([0,0,0,0,1,0,0,0,0]).reshape(1,9),
            'DS' :np.array([0,0,0,0,0,1,0,0,0]).reshape(1,9),
            'A' :np.array([0,0,0,0,0,0,1,0,0]).reshape(1,9),
            'D' :np.array([0,0,0,0,0,0,0,1,0]).reshape(1,9),
            '' :np.array([0,0,0,0,0,0,0,0,1]).reshape(1,9),
            
        }

move_decode ={
            0:'W',        
            1:'AW',
            2:'DW',
            3:'S',
            4:'AS',
            5:'DS',
            6:'A',
            7:'D',
            8:''
        }


def encode_movement(move):
    try:
        return move_encode[''.join(move)]
    except KeyError:
        return move_encode[''.join([])]


def model_sub_op(inputLayer):
    inputLayer = Convolution2D(3 , kernel_size=(5,5), strides=(2,2) , padding='valid' , activation='relu')(inputLayer)
    inputLayer = Convolution2D(24 , kernel_size=(5,5), strides=(2,2) , padding='valid' , activation='relu')(inputLayer)
    inputLayer = Convolution2D(36 , kernel_size=(5,5), strides=(2,2) , padding='valid' , activation='relu')(inputLayer)
    inputLayer = Convolution2D(48 , kernel_size=(3,3), strides=(1,1) , padding='valid' , activation='relu')(inputLayer)
    inputLayer = Convolution2D(64 , kernel_size=(3,3), strides=(1,1) , padding='valid' , activation='relu')(inputLayer)
    return inputLayer


def bifocal_nvidia(shape , margin):
    right_input = Input(shape=(shape[0],shape[1]-margin,shape[2]) , name='right_input')
    left_input = Input(shape=(shape[0],shape[1]-margin,shape[2]) , name='left_input')
    
    right_output = model_sub_op(right_input)
    left_output = model_sub_op(left_input)
    
    dense_input = concatenate([right_output , left_output] , axis=-1) 
    # concatenate is the functional interface and Conacatenate is the sequential interface
    dense_layers = Flatten()(dense_input)
    dense_layers = Dense(256, activation='relu')(dense_layers)
    dense_layers = Dropout(0.5)(dense_layers)
    dense_layers = Dense(128, activation='relu')(dense_layers)
    dense_layers = Dropout(0.25)(dense_layers)
    dense_layers = Dense(64, activation='relu')(dense_layers)
    dense_layers = Dropout(0.10)(dense_layers)
    dense_layers = Dense(32, activation='relu')(dense_layers)
    dense_layers = Dropout(0.5)(dense_layers)
    dense_layer_output = Dense(len(move_decode), activation='softmax')(dense_layers)
    
    model = Model(inputs=[right_input , left_input] , outputs=dense_layer_output)
    
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
    from datacollect import load_train_data , shape , margin
    X_train , Y_train = load_train_data()
    model = load_model_weights('v2_nfs_ai')
    for i in range(100):
        model.fit({'right_input':X_train[0] , 'left_input':X_train[1]}, Y_train, 
              batch_size=2000,nb_epoch=10, verbose=1)
        save_model(model , "v2_nfs_ai")


    