# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 12:36:24 2018

@author: Marco
"""


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
import os


from P2_funciones import play_rec_nidaqmx
from P2_funciones import play_rec
from P2_funciones import signalgen
from P2_funciones import sincroniza_con_trigger
from P2_funciones import fft_power_density


params = {'legend.fontsize': 24,
         'figure.figsize': (14, 9),
         'axes.labelsize': 24,
         'axes.titlesize':24,
         'xtick.labelsize':24,
         'ytick.labelsize':24}
pylab.rcParams.update(params)


#%%
    

# Genero matriz de señales: ejemplo de barrido en frecuencias en el canal 0
fs_out = int(4*44100)
fs_in = 10000
duracion = 1
muestras = int(fs_out*duracion)
input_channels = 1
output_channels = 1
amplitud = 1
frec_ini = 20
frec_fin = 21000
pasos = 200
delta_frec = (frec_fin-frec_ini)/(pasos+1)
data_out = np.zeros([pasos,muestras,output_channels])
frecs = np.array([])
for i in range(pasos):
    parametros_signal = {}
    amp = amplitud
    fr = frec_ini + i*delta_frec
    duration = duracion
    
    output_signal = signalgen('sine',fr,amp,duration,fs_out)
    data_out[i,:,0] = output_signal
    frecs = np.append(frecs,fr)
    


# Realiza medicion
offset_correlacion = 0#int(fs*(1))
steps_correlacion = 0#int(fs*(1))
#data_in, retardos = play_rec(fs,input_channels,data_out,'no',offset_correlacion,steps_correlacion)
data_in, retardos = play_rec_nidaqmx(fs_in,fs_out,input_channels,data_out,'no',offset_correlacion,steps_correlacion)

#%%

carpeta_salida = 'Aliasing'
subcarpeta_salida = '1'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
    
if not os.path.exists(os.path.join(carpeta_salida,subcarpeta_salida)):
    os.mkdir(os.path.join(carpeta_salida,subcarpeta_salida))  

frec_max_out = np.array([])
frec_max_in = np.array([])

# Le agegamos un offset por si tomo como maximo la frecuencia cero
offset_frec = 1

for i in range(data_out.shape[0]):
    
    freq_out, psdx_out = fft_power_density(data_out[i,:,0],fs)
    freq_in, psdx_in = fft_power_density(data_in[i,:,0],fs_in)
    
    offset_ind = int(np.ceil(offset_frec/(freq_in[1]-freq_in[0])))
    
    ind_out = np.argmax(psdx_out)
    frec_max = freq_out[int(ind_out)]
    frec_max_out = np.append(frec_max_out,frec_max)

    ind_in = np.argmax(psdx_in[offset_ind:])
    frec_max = freq_in[offset_ind+int(ind_in)]
    frec_max_in = np.append(frec_max_in,frec_max)  
    
np.save(os.path.join(carpeta_salida,subcarpeta_salida, 'frec_max_out'),frec_max_out)
np.save(os.path.join(carpeta_salida,subcarpeta_salida, 'frec_max_in'),frec_max_in)


#%%

carpeta_salida = 'Aliasing'
subcarpeta_salida = '1'

frec_max_out = np.load(os.path.join(carpeta_salida,subcarpeta_salida, 'frec_max_out.npy'))
frec_max_in = np.load(os.path.join(carpeta_salida,subcarpeta_salida, 'frec_max_in.npy'))


fig = plt.figure(dpi=250)
ax = fig.add_axes([.15, .15, .75, .8])
ax.plot(frec_max_out,frec_max_in,'o')
ax.set_xlabel('Frecuencia enviada [Hz]')
ax.set_ylabel('Frecuencia adquirida [Hz]')
ax.grid(linestyle='--',linewidth=0.5)
figname = os.path.join(carpeta_salida,subcarpeta_salida, 'aliasing.png')
fig.savefig(figname, dpi=300)  
plt.close(fig) 


#%%

