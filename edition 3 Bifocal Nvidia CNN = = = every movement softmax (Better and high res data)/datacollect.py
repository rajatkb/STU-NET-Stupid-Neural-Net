# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 06:43:36 2018

@author: RAJAT
"""
import numpy as np

shape = (200 , 400 , 1) # height * length of the input image to the neural net
margin = 200            # margin for angle of sight of both camera

def load_train_data(name):
    X_train=np.load(name+'_X.npy')
    Y_train=np.load(name+'_Y.npy')
    return X_train , Y_train
    
   
if __name__ == '__main__':
    import time
    import cv2
    
    from getkeys import key_check
    from grabscreen import grab_screen
    from model import encode_movement
    
    moves_count={'W':0 , 'AW':0 , 'DW':0 , 'D':0 , 'A':0}
    
    def verify_differnce(move , diff):
        move = ''.join(move)
        if(move == 'W'):
            if(moves_count['W'] - moves_count['AW'] > diff):
                return False
            elif(moves_count['W'] - moves_count['DW'] > diff):
                return False
            elif(moves_count['W'] - moves_count['D'] > diff*100):
                return False
            if(moves_count['W'] - moves_count['A'] > diff*100):
                return False
            else:
                return True
        else:
            return True
        
    def train_data_collect(view , move):
        right_view = view[:,0:shape[1]-margin]
        left_view = view[:,margin:]
        X_train[0].append(right_view.reshape(right_view.shape[0],right_view.shape[1],1))
        X_train[1].append(left_view.reshape(left_view.shape[0],left_view.shape[1],1))
        encoded_move = encode_movement(move)
        
        mlist=''.join(move)
        moves_count.setdefault(mlist , 0)
        moves_count[mlist]+=1
        
        Y_train.append(encoded_move)

        print(moves_count," ( ",len(X_train[0])," ) ")
        
        
    def save_train_data(name):
        X= np.array(X_train)
        Y= np.array(Y_train).reshape(-1,Y_train[0].shape[1])
        np.save(name+'_X.npy',X)
        np.save(name+'_Y.npy',Y)
        return X,Y
    #need to change how the data is collected
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
    
    frame = 0
    skip = 2
    
    while(True):
        current_view = grab_screen([5 , 20 , 800 , 600])
        reduced_view = cv2.resize(current_view , (shape[1],shape[0]) , interpolation= cv2.INTER_AREA)
        instant_move = key_check()
        frame+=1
        if(frame % skip == 0  and verify_differnce(instant_move , 2000)):
            train_data_collect(reduced_view , instant_move)
            frame=0
        cv2.imshow('rv',reduced_view)
        if cv2.waitKey(1) == 27 :
            break
        
    print(moves_count," ( ",len(X_train[0])," ) ")
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    save_train_data("second_batch")