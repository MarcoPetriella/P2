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
from P2_funciones import callback_pid
from P2_funciones import save_to_np_file
from P2_funciones import load_from_np_file
import time
import matplotlib.animation as animation




##
carpeta_salida = 'PID2'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
         
# Variables 
ai_channels = [1,2]
buffer_chunks = 500
ai_samples = 500
ai_samplerate = 50000
do_channel = [0]
initial_do_duty_cycle = 0.5
initial_do_frequency = 4000
setpoint = 4.6
kp = 0.1
ki = 0
kd = 0
isteps = 20
path_data_save = os.path.join(carpeta_salida,'experimento')
callback_pid_variables = {}

##
parametros = {}
parametros['buffer_chunks'] = buffer_chunks
parametros['ai_samples'] = ai_samples
parametros['ai_samplerate'] = ai_samplerate
parametros['ai_device'] = 'Dev1/ai'
parametros['ai_channels'] = ai_channels  
parametros['do_device'] = 'Dev1/ctr'
parametros['do_channel'] = do_channel  
parametros['initial_do_duty_cycle'] = initial_do_duty_cycle
parametros['initial_do_frequency'] = initial_do_frequency
parametros['setpoint'] = setpoint
parametros['pid_constants'] = [kp,ki,kd,isteps]
parametros['save_raw_data'] = True
parametros['save_processed_data'] = True
parametros['path_data_save'] = path_data_save
parametros['show_plot'] = True
parametros['callback_pid'] = callback_pid    
parametros['callback_pid_variables'] = callback_pid_variables

parametros['sub_chunk_save'] = 25
parametros['sub_chunk_plot'] = 25
parametros['nbr_buffers_plot'] = 5
parametros['plot_rate_hz'] = 13

pid_daqmx(parametros)
