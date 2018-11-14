# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:15:03 2018

@author: Marco
"""
#import matplotlib
#matplotlib.use('GTKAgg') 
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
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import os
from P2_funciones import pid_daqmx
from P2_funciones import save_to_np_file
from P2_funciones import load_from_np_file
import time
import matplotlib.animation as animation


def callback_pid(i, input_buffer, output_buffer_duty_cycle, output_buffer_pid_terminos, output_buffer_mean_data, output_buffer_error_data, output_buffer_pid_constants, buffer_chunks, sample_period ,callback_pid_variables):
    
        """
        # Variables de entrada fijas:
        # --------------------------
        # i: posicion actual en el input buffer. Recordar que i [0,buffer_chunks]
        # input_buffer: np.array [buffer_chunks, samples, ai_nbr_channels]
        # output_buffer_duty_cycle: np.array [buffer_chunks]. La i-posicion es la vieja, y se la calcula en el callback (output_buffer_duty_cycle_i)
        # output_buffer_pid_terminos: np.array [buffer_chunks,3] [buffer_chunks,termino_p termino_i termino_d]. La i-posicion es la vieja, y se la calcula en el callback
        #   termino_p: termino multiplicativo
        #   termino_i: termino integral
        #   termino_d: termino derivativo
        # output_buffer_mean_data: np.array [buffer_chunks,ai_nbr_channels]. La i-posicion es la actual. La columna 0 se utiliza para el PID.
        # output_buffer_error_data: np.array [buffer_chunks]. La i-posicion es la actual: output_buffer_mean_data[i,0] - lsetpoint[0]. Importante: se toma con el valor del setpoint del slider!
        # buffer_chunks: Cantidad de chunks del buffer
        # output_buffer_pid_constants: np.array [buffer_chunks, 5] - > [buffer_chunks, setpoint kp ki kd isteps]. La i-posicion es la actual. Importante: se toma con el valor del setpoint del slider!
        #   setpoint: Valor de tensión del setpoint
        #   kp: constante multiplicativa del PID
        #   ki: constante integral del PID
        #   kd: constante derivativa del PID
        #   isteps: cantidad de pasos para atrás utilizados para el termino integral
        #
        #
        # Variables de entrada del usuario:
        # --------------------------------
        # callback_pid_variables: lista con variables puestas por el usuario
        #   Ejemplo:
        #   variable0 = callback_pid_variables[0]
        #   variable1 = callback_pid_variables[1]
        #
        #
        # Salidas (calculadas en el callback):
        # -------
        # output_buffer_duty_cycle_i: duty cycle calculado en el callback
        # output_buffer_error_data_i: error calculado en el callback
        # termino_p: termino multiplicativo que acompaña a kp
        # termino_i: termino integral que acompaña a ki
        # termino_d: termino derivatico que acompaña a kd
        # setpoint: Valor de tensión del setpoint
        # kp: constante multiplicativa del PID
        # ki: constante integral del PID
        # kd: constante derivativa del PID
        # isteps: cantidad de pasos para atrás utilizados para el termino integral        
        #####################################
        """
        
        # Valores maximos y minimos de duty cycle
        max_duty_cycle = 0.999
        min_duty_cycle = 0.001  
        
        # Cargo valores actuales
        setpoint = output_buffer_pid_constants[i,0]
        kp = output_buffer_pid_constants[i,1]
        ki = output_buffer_pid_constants[i,2]
        kd = output_buffer_pid_constants[i,3]
        isteps = int(output_buffer_pid_constants[i,4])
        output_buffer_error_data_i = output_buffer_error_data[i]
        mean_data_i = output_buffer_mean_data[i,0]
            
        # Paso anterior de buffer circular
        j = (i-1)%buffer_chunks
        
#        ## Ejemplo de cambio de parametros PID on the fly
#        setpoint = output_buffer_pid_constants[j,0] - 0.002
#        kd = output_buffer_pid_constants[j,3] + 0.002
#        output_buffer_error_data_i = mean_data_i - setpoint
                
        # isteps paso anterior de buffer circular
        k = (i-isteps)%buffer_chunks    
        
        # Algoritmo PID
        termino_p = output_buffer_error_data_i
        termino_d = output_buffer_error_data_i - output_buffer_error_data[j]
        termino_d = termino_d*sample_period
        
        # Termino integral (hay que optimizar esto)
        termino_i = 0
        if k >= i:
            termino_i = np.sum(output_buffer_error_data[k:buffer_chunks]) + np.sum(output_buffer_error_data[0:i])
        else:
            termino_i = np.sum(output_buffer_error_data[k:i])
        termino_i = termino_i*sample_period
        
        output_buffer_duty_cycle_i =  kp*termino_p + ki*termino_i + kd*termino_d
    
        # Salida de la función
        output_buffer_duty_cycle_i = min(output_buffer_duty_cycle_i,max_duty_cycle)
        output_buffer_duty_cycle_i = max(output_buffer_duty_cycle_i,min_duty_cycle)
        
        return output_buffer_duty_cycle_i, output_buffer_error_data_i, termino_p, termino_i, termino_d, setpoint, kp, ki, kd, isteps 






##
carpeta_salida = 'PID2'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
         
# Variables 
ai_channels = [4]
buffer_chunks = 2000
ai_samples = 500
ai_samplerate = 50000
do_channel = [0]
initial_do_duty_cycle = 0.5
initial_do_frequency = 2000
setpoint = 4.42
kp = 0.96
ki = 22.98
kd = 19.86
isteps = 60
path_data_save = os.path.join(carpeta_salida,'experimento6')
callback_pid_variables = {}



##
parametros = {}
parametros['buffer_chunks'] = buffer_chunks
parametros['ai_samples'] = ai_samples
parametros['ai_samplerate'] = ai_samplerate
parametros['ai_device'] = 'Dev2/ai'
parametros['ai_channels'] = ai_channels  
parametros['do_device'] = 'Dev2/ctr'
parametros['do_channel'] = do_channel  
parametros['initial_do_duty_cycle'] = initial_do_duty_cycle
parametros['initial_do_frequency'] = initial_do_frequency
parametros['setpoint'] = setpoint
parametros['pid_constants'] = [kp,ki,kd,isteps]
parametros['save_raw_data'] = False
parametros['save_processed_data'] = True
parametros['path_data_save'] = path_data_save
parametros['show_plot'] = True
parametros['callback_pid'] = callback_pid    
parametros['callback_pid_variables'] = callback_pid_variables

parametros['sub_chunk_save'] = 25
parametros['sub_chunk_plot'] = 25
parametros['nbr_buffers_plot'] = 2
parametros['plot_rate_hz'] = 10

pid_daqmx(parametros)


#%%

import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import os
import matplotlib.pylab as pylab


params = {'legend.fontsize': 14,
          'figure.figsize': (14, 9),
         'axes.labelsize': 24,
         'axes.titlesize':18,
         'font.size':18,
         'xtick.labelsize':24,
         'ytick.labelsize':24}
pylab.rcParams.update(params)

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


carpeta_salida = 'PID2'
samplerate = 50000

duty_cycle = load_from_np_file(os.path.join(carpeta_salida, 'experimento4_duty_cycle.bin'))
raw_data = load_from_np_file(os.path.join(carpeta_salida, 'experimento4_raw_data.bin'))
mean_data = load_from_np_file(os.path.join(carpeta_salida, 'experimento4_mean_data.bin'))

raw_data1 = raw_data[7000:8000,:,0]
t1 = np.arange(0,raw_data1.shape[1])/samplerate

fig = plt.figure(dpi=250)
ax = fig.add_axes([.15, .15, .75, .8])
ax.plot(t1*1000,np.transpose(raw_data1[0:200,:]))
ax.grid(linestyle='--',linewidth=0.5)
ax.grid(linestyle='--',linewidth=0.5)
ax.set_xlabel('Tiempo [ms]')
ax.set_ylabel('Tensión [V]')

figname = os.path.join(carpeta_salida, 'experimento4_raw_data_duty_cycle_fijo.png')
fig.savefig(figname, dpi=300)  
plt.close(fig) 


#plt.plot(np.transpose(raw_data1[:,:]))

vector = raw_data1[:,80]
bins = 300
[hist_frec,hist_tension] = np.histogram(vector,range=(3.5,4.5),bins=bins)
hist_tension = hist_tension[1:]

ind = np.argmax(hist_frec)
v_max = hist_tension[ind]
ind_r = np.argmin(np.abs(hist_frec - hist_frec[ind]/2))
v_ruido = np.abs(2*(hist_tension[ind] - hist_tension[ind_r]))  


fig = plt.figure(dpi=250)
ax = fig.add_axes([.15, .15, .75, .8])
ax.bar(hist_tension,hist_frec,align='center',width=0.003,linewidth=0.1,edgecolor=(0.9,0.9,0.9))
ax.set_xlim([3.8,4.3])
ax.grid(linestyle='--',linewidth=0.5)
ax.set_xlabel('Tensión [V]')
ax.set_ylabel('Frecuencia')

figname = os.path.join(carpeta_salida, 'experimento4_raw_data_ruido_duty_cycle_fijo.png')
fig.savefig(figname, dpi=300)  
plt.close(fig) 


###

vector = mean_data[7000:8000]
#vector = np.mean(raw_data1,axis=1)
bins = 400
[hist_frec,hist_tension] = np.histogram(vector,range=(4.25,4.75),bins=bins)
hist_tension = hist_tension[1:]

ind = np.argmax(hist_frec)
v_max = hist_tension[ind]
ind_r = np.argmin(np.abs(hist_frec - hist_frec[ind]/2))
v_ruido = np.abs(2*(hist_tension[ind] - hist_tension[ind_r]))  


fig = plt.figure(dpi=250)
ax = fig.add_axes([.15, .15, .75, .8])
ax.bar(hist_tension,hist_frec,align='center',width=0.0011,linewidth=0.1,edgecolor=(0.9,0.9,0.9))
ax.set_xlim([4.4,4.55])
ax.grid(linestyle='--',linewidth=0.5)
ax.set_xlabel('Tensión [V]')
ax.set_ylabel('Frecuencia')

figname = os.path.join(carpeta_salida, 'experimento4_mean_data_ruido_duty_cycle_fijo.png')
fig.savefig(figname, dpi=300)  
plt.close(fig) 