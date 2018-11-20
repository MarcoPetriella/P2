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
import struct
import array
import threading

arduino = serial.Serial('COM5', 115200, timeout=0.5)
arduino.set_buffer_size(rx_size = 8*1000, tx_size = 8*1000)

buffer_in_data = np.zeros([1000,10])
buffer_in_timestamp = [None]*1000

evento_salida = threading.Event()
 


def recibe():
    i = 0
    while not evento_salida.is_set():  
        
        try:
            rawString = arduino.readline()
            array_serial = np.fromstring(rawString, dtype=float, count=-1, sep=',')
            now = datetime.datetime.now()
            now = now.strftime("%Y-%m-%d %H:%M:%S.%f")  
            print(now,array_serial)
            
            if len(array_serial) == 10:
                buffer_in_data[i,:] = array_serial
                buffer_in_timestamp[i] = now
                i = i+1
                print('i: ' + str(i))
            
        except:
            print("Exiting")
            break    
        
        #print(arduino.inWaiting())
        while arduino.inWaiting() > 200:
            print('ava')
            arduino.readline()
        
        
def manda():   

    setpoint = 1.2
    kp = 0.1
    ki = 0.2
    kd = 0.3
    isteps = 20.   
    
    while not evento_salida.is_set():     
        time.sleep(1)
        try:
            setpoint += 0.1
            kp += 0.2
            ki += 0.7
            kd += 0.5
            isteps += 1.
            
            
            arduino.write(struct.pack('<fffff',setpoint,kp,ki,kd,isteps))
            print(arduino.writable())
            arduino.reset_output_buffer()
            #arduino.flush()    
        except:
            print("Error")
            


t1 = threading.Thread(target=recibe, args=[])
t2 = threading.Thread(target=manda, args=[])
t1.start()
t2.start()

while True:
    try: 
        time.sleep(0.2)
    except KeyboardInterrupt:
        evento_salida.set()
        

        print ('\n \n Medición interrumpida \n')
    
arduino.close()   



plt.plot(buffer_in_data[:,8])  
   
buffer_in_timestamp[0:200]         