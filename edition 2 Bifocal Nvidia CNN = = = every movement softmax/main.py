# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 02:56:02 2018

@author: RAJAT
"""

from getkeys import key_check
from grabscreen import grab_screen
from pykeyboard import PyKeyboard
from model import load_model_weights,move_decode
from datacollect import shape , margin

import cv2
import numpy as np
import time


kb = PyKeyboard()



def keypress(move):
    if move == 'W':
        kb.press_key('W')
        kb.release_key('S')
        kb.release_key('A')
        kb.release_key('D')
    elif move =='AW':
        kb.press_key('W')
        kb.release_key('S')
        kb.press_key('A')
        kb.release_key('D')
    elif move =='DW':
        kb.press_key('W')
        kb.release_key('S')
        kb.release_key('A')
        kb.press_key('D')
    elif move =='S':
        kb.release_key('W')
        kb.press_key('S')
        kb.release_key('A')
        kb.release_key('D')
    elif move =='AS':
        kb.release_key('W')
        kb.press_key('S')
        kb.press_key('A')
        kb.release_key('D')
    elif move =='DS':
        kb.release_key('W')
        kb.press_key('S')
        kb.release_key('A')
        kb.press_key('D')
    elif move =='A':
        kb.release_key('W')
        kb.release_key('S')
        kb.press_key('A')
        kb.release_key('D')
    elif move =='D':
        kb.release_key('W')
        kb.release_key('S')
        kb.release_key('A')
        kb.press_key('D')
    else:
        kb.release_key('W')
        kb.release_key('S')
        kb.release_key('A')
        kb.release_key('D')
  

 
if __name__ == '__main__':
    
    model = load_model_weights('v2_nfs_ai')
    
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
        right_view = reduced_view[:,0:shape[1]-margin]
        left_view = reduced_view[:,margin:]
        
        cv2.imshow('frame' , reduced_view)
        
        right_view = right_view.reshape(1,right_view.shape[0],right_view.shape[1],1)
        left_view = left_view.reshape(1,left_view.shape[0],left_view.shape[1],1)
        
        instant_move = key_check()
        res = model.predict({'right_input': right_view , 'left_input':left_view})
        output = move_decode[np.argmax(res)]
        
        keypress(output)
        
        print("press ",list(output) ," keycheck :" ,instant_move)
        if cv2.waitKey(1) == 27 :
            break
        
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)
#save_model(model , "v1_testing")