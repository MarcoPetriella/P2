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
#import nidaqmx.constants as constants
#import nidaqmx.stream_writers
import time




def callback_pid(i, input_buffer, output_buffer_duty_cycle, output_buffer_mean_data, output_buffer_error_data, buffer_chunks, setpoint, kp, ki, kd, callback_pid_variables):
    
        # Parametros de entrada del callback
        # variable0 = callback_pid_variables[0]
        # variable1 = callback_pid_variables[1]
        #
        #####################################
        
        # Valores maximos y minimos de duty cycle
        max_duty_cycle = 0.999
        min_duty_cycle = 0.001
    
        # Paso anterior de buffer circular
        j = (i-1)%buffer_chunks
        
        # Algoritmo PID
        output_buffer_duty_cycle_i = output_buffer_duty_cycle[j] + kp*output_buffer_error_data[i] + ki*np.sum(output_buffer_error_data) + kd*(output_buffer_error_data[i]-output_buffer_error_data[j])
                
        # Salida de la función
        output_buffer_duty_cycle_i = min(output_buffer_duty_cycle_i,max_duty_cycle)
        output_buffer_duty_cycle_i = max(output_buffer_duty_cycle_i,min_duty_cycle)
                
        return output_buffer_duty_cycle_i



##
carpeta_salida = 'PID2'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
         
# Variables 
ai_nbr_channels = 1
ai_channels = [1]
buffer_chunks = 200
ai_samples = 200
ai_samplerate = 50000
initial_do_duty_cycle = 0.5
initial_do_frequency = 4000
setpoint = 4.6
kp = 0.1
ki = 0.5
kd = 0.3
path_data_save = os.path.join(carpeta_salida,'experimento')
callback_pid_variables = {}

##

parametros = {}
parametros['buffer_chunks'] = buffer_chunks
parametros['ai_nbr_channels'] = ai_nbr_channels   
parametros['ai_channels'] = ai_channels  
parametros['ai_samples'] = ai_samples
parametros['ai_samplerate'] = ai_samplerate
parametros['initial_do_duty_cycle'] = initial_do_duty_cycle
parametros['initial_do_frequency'] = initial_do_frequency
parametros['setpoint'] = setpoint
parametros['pid_constants'] = [kp,ki,kd]
parametros['save_raw_data'] = False
parametros['save_processed_data'] = True
parametros['path_data_save'] = path_data_save
parametros['callback_pid'] = callback_pid    
parametros['callback_pid_variables'] = callback_pid_variables

parametros['sub_chunk_save'] = 100
parametros['sub_chunk_plot'] = 25
parametros['nbr_buffers_plot'] = 10
parametros['plot_rate_hz'] = 10

pid_daqmx(parametros)