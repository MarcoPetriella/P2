# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:01:23 2018

@author: Marco
"""
import numpy as np

def hola(a,b,c):
    

    
    return a+b




def algo(hola,c):
    
    a = 1
    b = 2
    
    out = hola(a,b,c)
    
    
    return out





algo(hola,3)

#%%

def hola(a,b,c):
    

    
    return a+b


parametros = {}
parametros['callback'] = hola
parametros['variables_callback'] = [1]

def algo(parametros):
    
    hola = parametros['callback']
    variables_callback = parametros['variables_callback']
    c = variables_callback[0]
    
    a = 1
    b = 2
    
    out = hola(a,b,c)
    
    
    return out


algo(parametros)

#%%


import numpy as np

a = np.array([1,2,3,4,5])


a = np.reshape(a,5,order='F')


#%%

import threading
import numpy as np

def funcion(parametros):
    
    variables = parametros['variables']   
    callback = parametros['callback']
    
    buffer = np.zeros(30)
    
    def producer_thread():
        for i in range(5):
            callback(i,buffer,variables)

    t1 = threading.Thread(target=producer_thread, args=[])
    
    t1.start()


variables = {}
variables[0] = np.zeros(10) 

def callback(i,buffer,variables):
    
    vector = variables[0]    
    vector[i] = i
    buffer[i] = i
    print(vector)
    
  
parametros = {}
parametros['callback'] = callback
parametros['variables'] = variables

    
funcion(parametros)

#%%
arr = np.array([1,2,3])

f_handle = open('hola1.npy', 'ab')
np.save(f_handle, arr*2)
f_handle.close()

a = np.load('hola1.npy')

f = open('hola1.npy', 'rb')
for _ in range(100):
    print(np.load(f))
    
    
#%%
    
    
    
def save_to_np_file(filename,arr):
    f_handle = open(filename, 'ab')
    np.save(f_handle, arr)
    f_handle.close()    
    

arr = np.zeros([10,5,2])

try:    
    save_to_np_file('a',arr)   
except:
    1
    


def load_from_np_file(filename):

    f = open(filename, 'rb')
    array = np.load(f)  
    while True:
        try:
            array = np.append(array,np.load(f),axis=0)
        except:
            break
    f.close()  

    return array      


array = load_from_np_file('a')

   
        
        
        #print(np.load(f))
    
    
    
variables = {}
variables[0] = ['si','no']

def funcion(variables):
    [vari,varo] = variables[0]
    print(varo)

    
funcion(variables)    
    
    
    
    
    
a = np.zeros(10)    
b = np.ones(5)

a = np.append(a,b)


b = 2+np.random.rand(100,20,2)

d = np.zeros([100,2])
    
c = np.mean(b[1,:,:],axis=0)

d[1,:] = c


#%%


from datetime import datetime
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
from random import randrange

x_data, y_data = [], []

figure = pyplot.figure()
line, = pyplot.plot_date(x_data, y_data, '-')

def update(frame):
    x_data.append(datetime.now())
    y_data.append(randrange(0, 100))
    line.set_data(x_data, y_data)
    figure.gca().relim()
    figure.gca().autoscale_view()
    return line,

animation = FuncAnimation(figure, update, interval=200)
pyplot.show()



#%%

from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randrange
import threading
import time
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from matplotlib.widgets import Button

def funcion():

    
    y_data = np.ones(1000)
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax1 = ax.twinx()
    line, = ax.plot(y_data, '-',color='blue')
    ax.set_ylim([0,10])
    ax1.set_ylim([0,1.2])
        
    ax.set_ylim([0,5])


    def exit_callback(event):
        global interrupt_exit
        interrupt_exit = [True]
        print(interrupt_exit[0])    
    
    def producer_thread():
                       
        global interrupt_exit
        interrupt_exit = [False]
        
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(exit_callback)          
        
        
    #    root = Tk()
    #    
    #    y_data = np.zeros(100)
    #    
    #    fig = pyplot.figure()
    #    ax = fig.add_subplot(111)
    #    line, = ax.plot(y_data, '-')
    #    graph = FigureCanvasTkAgg(fig, master=root)
    #    graph.get_tk_widget().pack(side="top",fill='both',expand=True)
        i = 0
        while interrupt_exit[0] is False:  
            #print(interrupt_exit[0])
            #ax.cla()
            line.set_ydata(np.ones(1000)*(i+1)*0.001)
            fig.canvas.draw()
            i = i+1
    #        ax.plot(range(10), dpts, marker='o', color='orange')
    #        graph.draw()
            time.sleep(0.02)
        
    #    def update(frame):
    #        while True:
    #            x_data.append(datetime.now())
    #            y_data.append(randrange(0, 100))
    #            line.set_data(x_data, y_data)
    #            figure.gca().relim()
    #            figure.gca().autoscale_view()
    #            return line,
    #    
    #    animation = FuncAnimation(figure, update, interval=200)
    #    pyplot.show()
    
    def consumer_thread():

        global interrupt_exit
        interrupt_exit = [False]
        
        i = 0
        while interrupt_exit[0] is False:  
            
            print('hola')
            i = i+1
            time.sleep(0.2)
            
    
    t1 = threading.Thread(target=producer_thread, args=[])
    t2 = threading.Thread(target=consumer_thread, args=[])
    
    t1.start()
    t2.start()




funcion()

#%%



import matplotlib.pyplot as plt
import time
import random
 
ysample = random.sample(range(-50, 50), 100)
 
xdata = []
ydata = []
 
plt.show()
 
axes = plt.gca()
axes.set_xlim(0, 100)
axes.set_ylim(-50, +50)
line, = axes.plot(xdata, ydata, 'r-')
 
for i in range(100):
    xdata.append(i)
    ydata.append(ysample[i])
    line.set_xdata(xdata)
    line.set_ydata(ydata)
    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.1)
 
# add this if you don't want the window to disappear at the end
plt.show()

#%%

import numpy as np

a = np.arange(100)
b = np.arange(10000)
sub_chunk_save = 10

i = 0
m = 100
    
#%%

if i%sub_chunk_save == 0:
    j = (i-sub_chunk_save)%len(a)  
    jj = (j+sub_chunk_save-1)%len(a) + 1
    
    n = (m-sub_chunk_save)%len(b)
    nn = (n+sub_chunk_save-1)%len(b) + 1
    
    print(j,jj,i)
    print(a[j:jj])    

#    print(b[n:nn])
 #   print(n,nn,i)

i = i+1
i = i%len(a)

#%%
    
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import threading
import datetime
import time
import matplotlib.pylab as pylab
from scipy import signal
from sys import stdout
import numpy.fft as fft
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
    
def exit_callback(event):
    print('hola')
    #interrupt_exit = [True]    
    
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
ax1 = ax.twinx()
#line1, = ax.plot(data_plot1, '-',color='blue')
#line2, = ax1.plot(data_plot2, '-',color='red')
ax.set_ylim([0,10])
ax1.set_ylim([0,1.2])

axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(exit_callback)   


#%%



fig = plt.figure(figsize=(6,3.5),dpi=250)
ax = fig.add_axes([.15, .45, .75, .5])  
ax1 = ax.twinx()


y_bar_semaphores = np.arange(1,6)
x_bar_semaphores = np.ones(5)*100
ax2 = fig.add_axes([.15, .03, .4, .3]) 
ax2.barh(y_bar_semaphores, x_bar_semaphores, align='center',
    color=(1, 1, 1, 0), edgecolor='black')   
ax2.set_xticks([])
ax2.axis('off')
ax2.text(-50,1,'Input buffer',fontsize=8,va='center')

#%%

ai_nbr_channels = 2
ai_samples = 1000

input_buffer = np.zeros([100,ai_samples,ai_nbr_channels])


medicion = np.zeros([ai_nbr_channels,ai_samples])
medicion[1,:] = np.arange(0,ai_samples)

medicion = np.reshape(medicion,ai_nbr_channels*ai_samples,order='F')

for j in range(ai_nbr_channels):
    input_buffer[1,:,j] = medicion[j::ai_nbr_channels]  
    
    
input_buffer[1,:,0] 
#%%



def load_from_np_file(filename):

    f = open(filename, 'rb')
    array = np.load(f)  
    while True:
        try:
            array = np.append(array,np.load(f),axis=0)
        except:
            break
    f.close()  

    return array      


array = load_from_np_file('PID2\experimento_pid_constants.bin')
#array = load_from_np_file('PID2\experimento_duty_cycle.bin')

#array = load_from_np_file('PID2\experimento_pid_terminos.bin')


plt.plot(array[:,:])
plt.plot(array[210,:,0])

#%%

rects = plt.hist(range(10))

'%02d' % 2 + ' %'


previous = datetime.datetime.now()
now = datetime.datetime.now()
delta_ti = now - previous


delta_ti.total_seconds()

b = np.arange(5)

b[-5:]

axsetpoint = plt.axes([0.55, 0.27, 0.2, 0.03])

plt.axes(edgeli)

hl = plt.axhline(2)

hl.set_v



hola = [0]



hola = [1] + hola

#%%

ai_samples = 10000
ai_nbr_channels = 1
input_buffer = np.zeros([1000,ai_samples,ai_nbr_channels])


delta_t = np.array([])

for i in range(1000):
    
    medicion = []
    for i in range(ai_nbr_channels):
        medicion.append(i*5+np.random.rand(ai_samples))
    
    previous = datetime.datetime.now()
    medicion = np.asarray(medicion)
    for j in range(ai_nbr_channels):
        input_buffer[i,:,j] = medicion[j::ai_nbr_channels] 
    now = datetime.datetime.now()
    
    delta_ti = now - previous
    delta_ti = delta_ti.total_seconds()*1000   
    delta_t = np.append(delta_t,delta_ti)
    



plt.plot(input_buffer[5,:,1])


#%%

ai_samples = 10000
ai_nbr_channels = 1
input_buffer = np.zeros([1000,ai_samples])


delta_t = np.array([])

for i in range(1000):
    
    medicion = []
    medicion.append(np.random.rand(ai_samples))
    medicion.append(5+np.random.rand(ai_samples))    
    
    medicion = np.random.rand(ai_samples)
    
    previous = datetime.datetime.now()
    medicion = np.asarray(medicion)
    input_buffer[i,:] = medicion
    now = datetime.datetime.now()
    
    delta_ti = now - previous
    delta_ti = delta_ti.total_seconds()*1000   
    delta_t = np.append(delta_t,delta_ti)

#%%
    
a = np.zeros([2,3])

b = np.append(a,a, axis=0)

import pyqtgraph as pg
import numpy as np
x = np.arange(1000)
y = np.random.normal(size=(3, 1000))
plotWidget = pg.plot(title="Three plot curves")
for i in range(3):
    plotWidget.plot(x, y[i], pen=(i,3))
    
   
    #%%
    
buffer_chunks = 100
n_paso_anterior = 10
i = 10

k = (i-n_paso_anterior)%buffer_chunks    
print(k)

#

a = [True,True]
all(a)

while not all(a):
    print(1)
    
threads_exit_flags1 = []
threads_exit_flags1[0] = [False,False,False,False,False,False]


#%%



def hola(a,b):
    
    c = a+b
    
    return c

a = 1
b = 2

c = hola(a,b)