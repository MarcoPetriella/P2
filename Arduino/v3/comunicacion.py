# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:18:03 2018

@author: Marco
"""
#%%

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

arduino = serial.Serial('COM4', 2*9600, timeout=0.5)
arduino.set_buffer_size(rx_size = 8*1000, tx_size = 8*1000)

cant_chunk_rec = 1
sub_chunk_plot = 1
cant_variables= 11
buffer_in_data = np.zeros([10000,cant_variables])
buffer_in_timestamp = [None]*10000

setpoint = 2000
kp = 0.1
ki = 0.2
kd = 0.3
isteps = 20.   

evento_salida = threading.Event()
semaphore1 = threading.Semaphore(0) 


def recibe():
    i = 0
    while not evento_salida.is_set():  
        
        try:            
            rawString = arduino.read(4*cant_variables*cant_chunk_rec)
            array_serial = struct.unpack(cant_variables*cant_chunk_rec*'f',rawString)
            
            now = datetime.datetime.now()
            now = now.strftime("%Y-%m-%d %H:%M:%S.%f")  
            
            if len(array_serial) == cant_variables*cant_chunk_rec:
                buffer_in_data[i,:] = array_serial
                buffer_in_timestamp[i] = now
                i = i+1
                i = i%buffer_in_data.shape[0]
                semaphore1.release()
            
        except:
            print("Error en la lectura")
        
        while arduino.inWaiting() > 200:
            print('Limpiando buffer')
            arduino.readline()
        


        
def manda():   

    setpoint = 2000
    kp = 0.77
    ki = 0.2
    kd = 0.3
    isteps = 20.   
    
    while not evento_salida.is_set():     
        time.sleep(2)
        try:
             
            arduino.write(struct.pack('<fffff',setpoint,kp,ki,kd,isteps)) 
        except:
            print("Error en el envio")
            


def grafica():
    
    data = np.zeros(buffer_in_data.shape[0])
    i = 0
    while not evento_salida.is_set(): 

  
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
    

data = np.zeros(buffer_in_data.shape[0])  
        
fig = plt.figure(figsize=(7,3.7),dpi=250)
ax = fig.add_axes([.15, .15, .70, .70])  
line, = ax.plot(data, '.')      
ax.set_ylim([0,5000])

t1 = threading.Thread(target=recibe, args=[])
t2 = threading.Thread(target=manda, args=[])
t3 = threading.Thread(target=grafica, args=[])
t1.start()
t2.start()
t3.start()



#%%
evento_salida.set()
time.sleep(1)
arduino.close()

#%%

plt.plot(np.diff(buffer_in_data[:,10]),'.')
plt.ylim([0,100])

plt.plot(np.diff(buffer_in_data[:,8]),'.')
plt.ylim([0,300]) 

plt.plot(buffer_in_data[:,0],'.')
plt.ylim([0,5000])

plt.plot(buffer_in_data[:,4],'.')
plt.ylim([0,200])

plt.plot(buffer_in_data[:,4],'.')
plt.plot(buffer_in_data[:,5],'.')
plt.ylim([-1000,1000])

plt.plot(buffer_in_data[:,5],'-',alpha=0.6)
plt.plot(buffer_in_data[:,7],'-',alpha=0.6)
plt.ylim([-1000,1000])

plt.plot(buffer_in_data[:,6],'-',alpha=0.6)
plt.ylim([-1000,1000])

plt.plot(np.diff(buffer_in_data[:,5])*420)
plt.plot(buffer_in_data[:,7])
plt.ylim([-20000,20000])