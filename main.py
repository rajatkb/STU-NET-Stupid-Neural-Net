# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 02:56:02 2018

@author: RAJAT
"""

from getkeys import key_check
from grabscreen import grab_screen
from pykeyboard import PyKeyboard
from model import load_model_weights,move_decode
from datacollect import shape

import cv2
import numpy as np
import time


kb = PyKeyboard()

def forward():
    kb.press_key('W')
    kb.release_key('A')
    kb.release_key('D')
    kb.release_key('S')

def backward(): 
    kb.release_key('W')
    kb.press_key('S')
    

def right():
    kb.press_key('D')
    kb.release_key('A')
    kb.release_key('S')
    
def left():
    kb.press_key('A')
    kb.release_key('D')
    kb.release_key('S')
    
def nothing():
    kb.release_key('W')
    kb.release_key('A')
    kb.release_key('D')
    kb.release_key('S')
    
key_press={
        'W': forward,
        'S': backward,
        'A': left,
        'D': right,
        '.': nothing
        }
  

 
if __name__ == '__main__':
    
    model = load_model_weights('v1_nfs_ai')
    
    print("Starting !!!")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("3")
    time.sleep(1)
    print("4")
    print("GO go go go !!!!")
    
    
    
    while(True):
        current_view = grab_screen([0 , 0 , 800 , 600])
        reduced_view = cv2.resize(current_view , (shape[1],shape[0]) , interpolation= cv2.INTER_AREA)
        reduced_view = reduced_view.reshape(1,shape[0],shape[1],1)
        instant_move = key_check()
        res = model.predict(reduced_view)
        move = move_decode[np.argmax(res[0])]
        print("press ",move ," keycheck :" ,key_check())
        cv2.imshow('frame' , current_view)
        key_press[move]()
        if cv2.waitKey(1) == 27 :
            break
        
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)
#save_model(model , "v1_testing")
