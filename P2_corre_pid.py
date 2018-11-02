# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:15:03 2018

@author: Marco
"""

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
import nidaqmx
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import os
from P2_funciones import pid_daqmx
from P2_funciones import save_to_np_file
from P2_funciones import load_from_np_file
import nidaqmx.constants as constants
import nidaqmx.stream_writers
import time




def callback(i, input_buffer, output_buffer_duty_cycle, output_buffer_frequency, buffer_chunks, initial_do_duty_cycle, initial_do_frequency, callback_variables):
    
        # Parametros de entrada del callback
        vector_mean_ch1 = callback_variables[0]
        path_vector_mean_ch1 = callback_variables[1]
        path_duty_cycle = callback_variables[2]
        path_input_buffer = callback_variables[3]
        set_point = callback_variables[4]
        vector_error = callback_variables[5]
        constantes_pid = callback_variables[6]
        kp = constantes_pid[0]
        ki = constantes_pid[1]
        kd = constantes_pid[2]
        #####################################
        
        # Valores maximos y minimos de duty cycle
        max_duty_cycle = 0.999
        min_duty_cycle = 0.001
    
        # Leo del buffer y calculo el error
        ch1_array = input_buffer[i,:,0]          
        mean_ch1 = np.mean(ch1_array)
        vector_mean_ch1[i] = mean_ch1
        error = mean_ch1 - set_point
        vector_error[i] = error

        # Paso anterior de buffer circular
        j = (i-1)%buffer_chunks
        
        # Algoritmo PID
        output_buffer_duty_cycle_i = output_buffer_duty_cycle[j] + kp*error + ki*np.sum(vector_error) + kd*(vector_error[i]-vector_error[j])
        
        # Salidas de la función
        output_buffer_duty_cycle_i = min(output_buffer_duty_cycle_i,max_duty_cycle)
        output_buffer_duty_cycle_i = max(output_buffer_duty_cycle_i,min_duty_cycle)
        output_buffer_frequency_i = initial_do_frequency
        
        # Para grabar
        if i == buffer_chunks-1:
           save_to_np_file(path_duty_cycle,output_buffer_duty_cycle)               
           save_to_np_file(path_vector_mean_ch1,vector_mean_ch1)   
           save_to_np_file(path_input_buffer,input_buffer)   

        return output_buffer_duty_cycle_i, output_buffer_frequency_i



##
carpeta_salida = 'PID'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
         
# Variables 
ai_nbr_channels = 1
ai_channels = [1,2]
buffer_chunks = 100
ai_samples = 1000
ai_samplerate = 50000
initial_do_duty_cycle = 0.5
initial_do_frequency = 200

# Variables Callback
vector_mean_ch1 = np.zeros(buffer_chunks) 
path_vector_mean_ch1 = os.path.join(carpeta_salida,'vector_media.npy')
path_duty_cycle = os.path.join(carpeta_salida,'vector_duty_cycle.npy')
path_input_buffer = os.path.join(carpeta_salida,'input_buffer.npy')

# Variables Callback PID
set_point = 4.6
vector_error = np.zeros(buffer_chunks)
kp = 0.1
ki = 0.5
kd = 0.3
constantes_pid = [kp,ki,kd]

##
callback_variables = {}
callback_variables[0] = vector_mean_ch1
callback_variables[1] = path_vector_mean_ch1
callback_variables[2] = path_duty_cycle
callback_variables[3] = path_input_buffer
callback_variables[4] = set_point
callback_variables[5] = vector_error
callback_variables[6] = constantes_pid


parametros = {}
parametros['buffer_chunks'] = buffer_chunks
parametros['ai_nbr_channels'] = ai_nbr_channels   
parametros['ai_channels'] = ai_channels  
parametros['ai_samples'] = ai_samples
parametros['ai_samplerate'] = ai_samplerate
parametros['initial_do_duty_cycle'] = initial_do_duty_cycle
parametros['initial_do_frequency'] = initial_do_frequency
parametros['callback'] = callback    
parametros['callback_variables'] = callback_variables

input_buffer, output_buffer_duty_cycle, output_buffer_frequency = pid_daqmx(parametros)