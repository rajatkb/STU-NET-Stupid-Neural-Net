# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 06:43:36 2018

@author: RAJAT
"""
import numpy as np


shape = (120 , 160 , 1) # height * length of the input image to the neural net
margin = 40            # margin for angle of sight of both camera

def load_train_data():
    X_train=np.load('X.npy')
    Y_train=np.load('Y.npy')
    return X_train , Y_train
    
   
if __name__ == '__main__':
    import time
    import cv2
    
    from getkeys import key_check
    from grabscreen import grab_screen
    from model import encode_movement
    
    def train_data_collect(view , move):
        right_view = view[:,0:shape[1]-margin]
        left_view = view[:,margin:]
        X_train[0].append(right_view.reshape(right_view.shape[0],right_view.shape[1],1))
        X_train[1].append(right_view.reshape(left_view.shape[0],right_view.shape[1],1))
        encoded_move = encode_movement(move)
        Y_train.append(encoded_move)
        print(" ( ",len(X_train[0])," , ",len(X_train[1])," , ", len(Y_train)," ) move:" ,move ," encoded move:" , encoded_move )
    
    def save_train_data():
        X= np.array(X_train)
        Y= np.array(Y_train).reshape(-1,Y_train[0].shape[1])
        np.save('X.npy',X)
        np.save('Y.npy',Y)
        return X,Y

    X_train,Y_train=([[],[]],[])
    
    print("Starting !!!")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("3")
    time.sleep(1)
    print("4")
    print("GO go go go RECORDING STARTED!!!!")
    
    
    
    while(True):
        current_view = grab_screen([5 , 20 , 800 , 600])
        reduced_view = cv2.resize(current_view , (shape[1],shape[0]) , interpolation= cv2.INTER_AREA)
        instant_move = key_check()
        train_data_collect(reduced_view , instant_move)
        cv2.imshow('rv',reduced_view)
        if cv2.waitKey(1) == 27 :
            break
        
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    save_train_data()