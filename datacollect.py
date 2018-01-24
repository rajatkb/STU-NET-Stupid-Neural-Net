# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 06:43:36 2018

@author: RAJAT
"""
import numpy as np


shape = (60 , 80 , 1) # height * length of the input image to the neural net


def load_train_data():
    try:
        X_train=np.load('X.npy')
        Y_train=np.load('Y.npy')
        return X_train , Y_train
    except:
        return ([],[])
    
   
if __name__ == '__main__':
    import time
    import cv2
    
    from getkeys import key_check
    from grabscreen import grab_screen
    from model import encode_movement
    
    def train_data_collect(view , move):
        X_train.append(view.reshape(shape[0],shape[1],1))
        Y_train.append(encode_movement(move))
    
    def save_train_data():
        X= np.array(X_train)
        Y= np.array(Y_train).reshape(-1,5)
        np.save('X.npy',X)
        np.save('Y.npy',Y)
        return X,Y

    X_train,Y_train=load_train_data()
    
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
        current_view = grab_screen([0 , 0 , 800 , 600])
        reduced_view = cv2.resize(current_view , (shape[1],shape[0]) , interpolation= cv2.INTER_AREA)
        instant_move = key_check()
        train_data_collect(reduced_view , instant_move)
        cv2.imshow('frame',current_view)
        print(" ( ",len(X_train), len(Y_train)," ) ")
        if cv2.waitKey(1) == 27 :
            break
        
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    save_train_data()