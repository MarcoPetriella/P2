import nidaqmx
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import os
from P2_funciones import cross_correlation_using_fft

#%%
num_samples = 50000
samplerate = 50000
#t = np.arange(0,num_samples/samplerate,1/samplerate)
t = np.linspace(0,(num_samples-1)/samplerate,num_samples)

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai2:5",max_val=2., min_val=-2.,name_to_assign_to_channel="ch1",terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)#, "Voltage")#,AIVoltageUnits.Volts)#,max_val=10., min_val=-10.)
    #task.ai_channels.add_ai_voltage_chan("Dev1/ai4")
    #task.ai_channels.ai_impedance  #(nidaqmx.constants.Impedance1.ONE_M_OHM)# =  # = nidaqmx.constants.Impedance1.ONE_M_OHM
    
    task.timing.cfg_samp_clk_timing(samplerate,samps_per_chan=num_samples)

    b=task.read(number_of_samples_per_channel=num_samples)


medicion = np.asarray(b)

#%%

# Primero calibramos los canales con una cuadrada
carpeta_salida = 'Tiempointercanal'
subcarpeta_salida = 'cuadrada'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
    
if not os.path.exists(os.path.join(carpeta_salida,subcarpeta_salida)):
    os.mkdir(os.path.join(carpeta_salida,subcarpeta_salida))


np.save(os.path.join(carpeta_salida,subcarpeta_salida, 'medicion'),medicion)

medicion = np.transpose(medicion)
plt.plot(medicion)



#%%
num_samples = 50000
samplerate = 50000
#t = np.arange(0,num_samples/samplerate,1/samplerate)
t = np.linspace(0,(num_samples-1)/samplerate,num_samples)

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai2:5",max_val=2., min_val=-2.,name_to_assign_to_channel="ch1",terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)#, "Voltage")#,AIVoltageUnits.Volts)#,max_val=10., min_val=-10.)
    #task.ai_channels.add_ai_voltage_chan("Dev1/ai4")
    #task.ai_channels.ai_impedance  #(nidaqmx.constants.Impedance1.ONE_M_OHM)# =  # = nidaqmx.constants.Impedance1.ONE_M_OHM
    
    task.timing.cfg_samp_clk_timing(samplerate,samps_per_chan=num_samples)

    b=task.read(number_of_samples_per_channel=num_samples)


medicion = np.asarray(b)

#%%

# Primero calibramos los canales con una cuadrada
carpeta_salida = 'Tiempointercanal'
subcarpeta_salida = 'rampa_2000hz'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
    
if not os.path.exists(os.path.join(carpeta_salida,subcarpeta_salida)):
    os.mkdir(os.path.join(carpeta_salida,subcarpeta_salida))


np.save(os.path.join(carpeta_salida,subcarpeta_salida, 'medicion'),medicion)

medicion = np.transpose(medicion)
plt.plot(medicion[:,0],'.',label='CH2')
plt.plot(medicion[:,1],'.',label='CH3')
plt.plot(medicion[:,2],'.',label='CH4')
plt.plot(medicion[:,3],'.',label='CH5')

plt.legend()


#%%
num_samples = 50000
samplerate = 50000
#t = np.arange(0,num_samples/samplerate,1/samplerate)
t = np.linspace(0,(num_samples-1)/samplerate,num_samples)

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai2:3",max_val=2., min_val=-2.,name_to_assign_to_channel="ch1",terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)#, "Voltage")#,AIVoltageUnits.Volts)#,max_val=10., min_val=-10.)
    #task.ai_channels.add_ai_voltage_chan("Dev1/ai4")
    #task.ai_channels.ai_impedance  #(nidaqmx.constants.Impedance1.ONE_M_OHM)# =  # = nidaqmx.constants.Impedance1.ONE_M_OHM
    
    task.timing.cfg_samp_clk_timing(samplerate,samps_per_chan=num_samples)

    b=task.read(number_of_samples_per_channel=num_samples)


medicion = np.asarray(b)

#%%

# Primero calibramos los canales con una cuadrada
carpeta_salida = 'Tiempointercanal'
subcarpeta_salida = 'rampa_2000hz_2ch'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
    
if not os.path.exists(os.path.join(carpeta_salida,subcarpeta_salida)):
    os.mkdir(os.path.join(carpeta_salida,subcarpeta_salida))


np.save(os.path.join(carpeta_salida,subcarpeta_salida, 'medicion'),medicion)

medicion = np.transpose(medicion)
plt.plot(medicion[:,0],'.',label='CH2')
plt.plot(medicion[:,1],'.',label='CH3')

plt.legend()


#%%
num_samples = 50000
samplerate = 50000
#t = np.arange(0,num_samples/samplerate,1/samplerate)
t = np.linspace(0,(num_samples-1)/samplerate,num_samples)

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai2:6",max_val=2., min_val=-2.,name_to_assign_to_channel="ch1",terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)#, "Voltage")#,AIVoltageUnits.Volts)#,max_val=10., min_val=-10.)
    #task.ai_channels.add_ai_voltage_chan("Dev1/ai4")
    #task.ai_channels.ai_impedance  #(nidaqmx.constants.Impedance1.ONE_M_OHM)# =  # = nidaqmx.constants.Impedance1.ONE_M_OHM
    
    task.timing.cfg_samp_clk_timing(samplerate,samps_per_chan=num_samples)

    b=task.read(number_of_samples_per_channel=num_samples)


medicion = np.asarray(b)

#%%

# Primero calibramos los canales con una cuadrada
carpeta_salida = 'Tiempointercanal'
subcarpeta_salida = 'rampa_2000hz_5ch'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
    
if not os.path.exists(os.path.join(carpeta_salida,subcarpeta_salida)):
    os.mkdir(os.path.join(carpeta_salida,subcarpeta_salida))


np.save(os.path.join(carpeta_salida,subcarpeta_salida, 'medicion'),medicion)

medicion = np.transpose(medicion)
plt.plot(medicion[:,0],'.',label='CH2')
plt.plot(medicion[:,1],'.',label='CH3')
plt.plot(medicion[:,2],'.',label='CH4')
plt.plot(medicion[:,3],'.',label='CH5')
plt.plot(medicion[:,4],'.',label='CH6')
plt.legend()


