# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:18:03 2018

@author: Marco
"""

import win32pipe, win32file,time,struct
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from multiprocessing import Process
import sys
import datetime 
import serial
import os


arduino = serial.Serial('COM5', 115200, timeout=0.1)


while True:
    
    try:
        line = arduino.readline()   
        #array_serial = np.fromstring(line, dtype=float, count=-1, sep=' ')
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S.%f")  
        print(now,line)
    
    except:
        print("Exiting")
        break        
    
    


arduino.close() 
    