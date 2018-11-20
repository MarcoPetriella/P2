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

arduino = serial.Serial('COM5', 2*9600, timeout=0.5)
arduino.set_buffer_size(rx_size = 8*1000, tx_size = 8*1000)

cant_variables= 10
sub_chunk_plot = 1
buffer_in_data = np.zeros([1000,cant_variables])
buffer_in_timestamp = [None]*1000

evento_salida = threading.Event()
semaphore1 = threading.Semaphore(0) 


def recibe():
    i = 0
    while not evento_salida.is_set():  
        
        try:
            rawString = arduino.readline()
            array_serial = np.fromstring(rawString, dtype=float, count=-1, sep=',')
            
#            rawString = arduino.read(4*cant_variables)
#            array_serial = struct.unpack(cant_variables*'f',rawString)
            
            now = datetime.datetime.now()
            now = now.strftime("%Y-%m-%d %H:%M:%S.%f")  
            
            if len(array_serial) == cant_variables:
                buffer_in_data[i,:] = array_serial
                buffer_in_timestamp[i] = now
                i = i+1
                i = i%buffer_in_data.shape[0]
                semaphore1.release()

                
                #print(i,now,array_serial)
            
        except:
            print("Exiting")
            #break    
        
        #print(arduino.inWaiting())
        while arduino.inWaiting() > 200:
            print('ava')
            arduino.readline()
        

#        try:
#            setpoint = 0.1
#            kp = 0.2
#            ki = 0.7
#            kd = 0.5
#            isteps = 1.
#            
#            
#            arduino.write(struct.pack('<fffff',setpoint,kp,ki,kd,isteps))
#            print(arduino.writable())
##            arduino.reset_output_buffer()
##            arduino.flush()    
#        except:
#            print("Error")
        
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
            #print(arduino.writable())
#            arduino.reset_output_buffer()
#            arduino.flush()    
        except:
            print("Error")
            


def grafica():
    
    data = np.zeros(buffer_in_data.shape[0])
    i = 0
    while not evento_salida.is_set(): 


#def grafica():
#    global data,j

        
        semaphore1.acquire()
            
        if not i%sub_chunk_plot:
            
            j = (i-sub_chunk_plot)%buffer_in_data.shape[0]  
            jj = (j+sub_chunk_plot-1)%buffer_in_data.shape[0]  + 1 
            #print(j,jj)
        
                    
            #data[0:-1] = data[1:]
            #data[data.shape[0]-1] = buffer_in_data[i,8]
            
            data = np.append(data,buffer_in_data[j:jj,8])
            data = data[sub_chunk_plot:]
            
            line.set_ydata(data)  
        
        i = i + 1
        i = i%buffer_in_data.shape[0]
        
        fig.canvas.draw_idle()
        #return line,
    

data = np.zeros(buffer_in_data.shape[0])  
#j = 0  
        
fig = plt.figure(figsize=(7,3.7),dpi=250)
ax = fig.add_axes([.15, .15, .70, .70])  
line, = ax.plot(data, '-')      
ax.set_ylim([0,5000])

t1 = threading.Thread(target=recibe, args=[])
t2 = threading.Thread(target=manda, args=[])
t3 = threading.Thread(target=grafica, args=[])
t1.start()
t2.start()
t3.start()

