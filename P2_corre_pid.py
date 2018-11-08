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
