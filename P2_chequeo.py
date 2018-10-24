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
from P2_funciones import play_rec1
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


fs_out = int(44100*4)
fs_in = fs_out

duracion = 0.5
muestras = int(fs_out*duracion)
input_channels = 2
output_channels = 2
amplitud = 0.5
frec_ini = 1023
frec_fin = 1023
pasos = 1
delta_frec = (frec_fin-frec_ini)/(pasos+1)
data_out = np.zeros([pasos,muestras,output_channels])

for i in range(pasos):
    parametros_signal = {}
    amp = amplitud
    fr = frec_ini + i*delta_frec
    duration = duracion
    
    if i == 0:
        output_signal = signalgen('sine',fr,amp,duration,fs_out)
        data_out[i,:,0] = output_signal#*np.arange(output_signal.shape[0])/output_signal.shape[0]
        
        output_signal = signalgen('sine',fr,amp,duration,fs_out)
        data_out[i,:,1] = output_signal


# Realiza medicion
offset_correlacion = 0#int(fs*(1))
steps_correlacion = 0#int(fs*(1))
data_in, retardos = play_rec1(fs_in,fs_out,input_channels,data_out,'si',offset_correlacion,steps_correlacion,dato='int16')


#%%

plt.plot(data_in[0,:,0])


#%%
    

# Genero matriz de señales: ejemplo de barrido en frecuencias en el canal 0
fs = int(4*44100)
fs_in = 10000
duracion = 1
muestras = int(fs*duracion)
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
    fs = fs
    amp = amplitud
    fr = frec_ini + i*delta_frec
    duration = duracion
    
    output_signal = signalgen('sine',fr,amp,duration,fs)
    data_out[i,:,0] = output_signal
    frecs = np.append(frecs,fr)
    


# Realiza medicion
offset_correlacion = 0#int(fs*(1))
steps_correlacion = 0#int(fs*(1))
#data_in, retardos = play_rec(fs,input_channels,data_out,'no',offset_correlacion,steps_correlacion)
data_in, retardos = play_rec_nidaqmx(fs_in,fs,input_channels,data_out,'no',offset_correlacion,steps_correlacion)

#%%

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
    frec_max = freq_out[offset_ind+int(ind_in)]
    frec_max_in = np.append(frec_max_in,frec_max)  
    


plt.plot(frec_max_out,frec_max_in,'o')






#%%

### Corrige retardo y grafica
#
#data_in, retardos = sincroniza_con_trigger(data_out, data_in) 

#%%
ch = 0
step = 0

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .15, .75, .8])
ax1 = ax.twinx()
ax.plot(np.transpose(data_in[step,:,ch]),'-',color='r', label='señal adquirida',alpha=0.5)
ax1.plot(np.transpose(data_out[step,:,ch]),'-',color='b', label='señal enviada',alpha=0.5)
ax.legend(loc=1)
ax1.legend(loc=4)
plt.show()


fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .15, .75, .8])
ax.hist(retardos/fs*1000, bins=100)
ax.set_title(u'Retardos')
ax.set_xlabel('Retardos [ms]')
ax.set_ylabel('Frecuencia')


#%%
### ANALISIS de la señal adquirida. Cheque que la señal adquirida corresponde a la enviada


ch_acq = 0
ch_send = 0
paso = 0

### Realiza la FFT de la señal enviada y adquirida

frec_acq,fft_acq = fft_power_density(data_in[paso,:,ch_acq],fs)
frec_send,fft_send = fft_power_density(data_out[paso,:,ch_send],fs)

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .12, .75, .8])
ax1 = ax.twinx()
ax.plot(frec_send,fft_send,'-' ,label='Frec enviada',alpha=0.7)
ax1.plot(frec_acq,fft_acq,'-',color='red', label=u'Señal adquirida',alpha=0.7)
ax.set_title(u'FFT de la señal enviada y adquirida')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Amplitud [a.u.]')
ax.legend(loc=1)
ax1.legend(loc=4)
plt.show()


fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .12, .75, .8])
ax.plot(frec_acq,fft_acq/fft_send,'-',color='red', label=u'Señal adquirida',alpha=0.7)
ax.set_xlim([0,23000])
ax.set_ylim([0,1e10])
ax.set_title(u'FFT de la señal enviada y adquirida')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Amplitud [a.u.]')
ax.legend(loc=1)
plt.show()
 

#%%

Is = 1.0*1e-12
Vt = 26.0*1e-3
n = 1.

Vd = np.linspace(-1,1,1000)
Id = Is*(np.exp(Vd/n/Vt)-1)

Rs = 100
Vs = 1
Ir = Vs/Rs - Vd/Rs


plt.plot(Vd,Id)
plt.plot(Vd,Ir)
