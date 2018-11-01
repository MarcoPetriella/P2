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




def callback(i, input_buffer, output_buffer_duty_cycle, output_buffer_frequency,chunks_buffer,initial_pid_duty_cycle,initial_pid_frequency, callback_variables):
    
        # Parametros de entrada del callback
        vector_mean_ch1 = callback_variables[0]
        path_vector_mean_ch1 = callback_variables[1]
        path_duty_cycle = callback_variables[2]
        
        # Parametros PID
        valor_esperado = callback_variables[3]
        vector_error = callback_variables[4]
        constantes_pid = callback_variables[5]
        kp = constantes_pid[0]
        ki = constantes_pid[1]
        kd = constantes_pid[2]
        #####################################
    
        ch1_array = input_buffer[i,:,0]          
        mean_ch1 = np.mean(ch1_array)
        vector_mean_ch1[i] = mean_ch1
        error = mean_ch1 - valor_esperado
        vector_error[i] = error

        j = i-1
        if j == -1:
            j == chunks_buffer-1
        
        output_buffer_duty_cycle_i = output_buffer_duty_cycle[j] + kp*error + ki*np.sum(vector_error) + kd*(vector_error[i]-vector_error[j])
        if output_buffer_duty_cycle_i > 0.99:
            output_buffer_duty_cycle_i = 0.99
        if output_buffer_duty_cycle_i < 0.01:
            output_buffer_duty_cycle_i = 0.01  

        if i == chunks_buffer-1:
           save_to_np_file(path_duty_cycle,output_buffer_duty_cycle)               
           save_to_np_file(path_vector_mean_ch1,vector_mean_ch1)   
                   
        output_buffer_frequency_i = initial_pid_frequency

        return output_buffer_duty_cycle_i, output_buffer_frequency_i



##

chunks_buffer = 100
initial_pid_duty_cycle = 0.5
initial_pid_frequency = 200
input_channels = 1

vector_mean_ch1 = np.zeros(chunks_buffer) 
valor_esperado = 4.6
vector_error = np.zeros(chunks_buffer)
kd = 0.1
ki = 0.5
kd = 0.3
constantes_pid = [kd,ki,kd]
path_vector_mean_ch1 = ''
path_duty_cycle = ''



callback_variables = {}
callback_variables[0] = vector_mean_ch1
callback_variables[1] = path_vector_mean_ch1
callback_variables[2] = path_duty_cycle
callback_variables[3] = valor_esperado
callback_variables[4] = vector_error
callback_variables[5] = constantes_pid


parametros = {}
parametros['chunks_buffer'] = chunks_buffer
parametros['ai_channels'] = input_channels    
parametros['ai_samples'] = 1000
parametros['ai_samplerate'] = 50000
parametros['initial_pid_duty_cycle'] = initial_pid_duty_cycle
parametros['initial_pid_frequency'] = initial_pid_frequency
parametros['callback'] = callback    
parametros['callback_variables'] = callback_variables

pid_daqmx(parametros)