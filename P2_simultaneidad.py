import nidaqmx
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import os
from P1_funciones import cross_correlation_using_fft


num_samples = 50000
samplerate = 50000
#t = np.arange(0,num_samples/samplerate,1/samplerate)
t = np.linspace(0,(num_samples-1)/samplerate,num_samples)

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai2:5",max_val=10., min_val=-10.,name_to_assign_to_channel="ch1",terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)#, "Voltage")#,AIVoltageUnits.Volts)#,max_val=10., min_val=-10.)
    #task.ai_channels.add_ai_voltage_chan("Dev1/ai4")
    #task.ai_channels.ai_impedance  #(nidaqmx.constants.Impedance1.ONE_M_OHM)# =  # = nidaqmx.constants.Impedance1.ONE_M_OHM
    
    task.timing.cfg_samp_clk_timing(samplerate,samps_per_chan=num_samples)

    b=task.read(number_of_samples_per_channel=num_samples)


medicion = np.asarray(b)

#%%
carpeta_salida = 'Simultaneidad'
subcarpeta_salida = 'seno_100hz'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
    
if not os.path.exists(os.path.join(carpeta_salida,subcarpeta_salida)):
    os.mkdir(os.path.join(carpeta_salida,subcarpeta_salida))  


np.save(os.path.join(carpeta_salida,subcarpeta_salida, 'medicion'),medicion)

#%%
carpeta_salida = 'Simultaneidad'
subcarpeta_salida = 'seno_1000hz'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
    
if not os.path.exists(os.path.join(carpeta_salida,subcarpeta_salida)):
    os.mkdir(os.path.join(carpeta_salida,subcarpeta_salida))  


np.save(os.path.join(carpeta_salida,subcarpeta_salida, 'medicion'),medicion)

#%%
carpeta_salida = 'Simultaneidad'
subcarpeta_salida = 'seno_5000hz'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
    
if not os.path.exists(os.path.join(carpeta_salida,subcarpeta_salida)):
    os.mkdir(os.path.join(carpeta_salida,subcarpeta_salida))  


np.save(os.path.join(carpeta_salida,subcarpeta_salida, 'medicion'),medicion)


#%%
carpeta_salida = 'Simultaneidad'
subcarpeta_salida = 'seno_100hz'

medicion = np.load(os.path.join(carpeta_salida,subcarpeta_salida, 'medicion.npy'))
medicion = np.transpose(medicion)
plt.plot(medicion)

#%%
carpeta_salida = 'Simultaneidad'
subcarpeta_salida = 'seno_1000hz'

medicion = np.load(os.path.join(carpeta_salida,subcarpeta_salida, 'medicion.npy'))
medicion = np.transpose(medicion)
plt.plot(medicion)

#%%
carpeta_salida = 'Simultaneidad'
subcarpeta_salida = 'seno_5000hz'

medicion = np.load(os.path.join(carpeta_salida,subcarpeta_salida, 'medicion.npy'))
medicion = np.transpose(medicion)
plt.plot(medicion)
#%%


