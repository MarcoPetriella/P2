import nidaqmx
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import os
from P2_funciones import cross_correlation_using_fft

#%%
num_samples = 10000
samplerate = 10000
#t = np.arange(0,num_samples/samplerate,1/samplerate)
t = np.linspace(0,(num_samples-1)/samplerate,num_samples)

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai2:3",max_val=10., min_val=-10.,name_to_assign_to_channel="ch1",terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)#, "Voltage")#,AIVoltageUnits.Volts)#,max_val=10., min_val=-10.)
    #task.ai_channels.add_ai_voltage_chan("Dev1/ai4")
    #task.ai_channels.ai_impedance  #(nidaqmx.constants.Impedance1.ONE_M_OHM)# =  # = nidaqmx.constants.Impedance1.ONE_M_OHM
    
    task.timing.cfg_samp_clk_timing(samplerate,samps_per_chan=num_samples)

    b=task.read(number_of_samples_per_channel=num_samples)


medicion = np.asarray(b)

#%%

# Primero calibramos los canales con una cuadrada
carpeta_salida = 'Settingtime'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
    


np.save(os.path.join(carpeta_salida, 'medicion_rampa_2000hz_10khz'),medicion)

medicion = np.transpose(medicion)
plt.plot(medicion[:,0],'.',label='CH2')
plt.plot(medicion[:,1],'.',label='CH3')
plt.legend()

v_ch1ch2 = 0.12
rampa = 8000
tiempo_ch1ch2 = v_ch1ch2/rampa

v_ch1ch1 = 0.160
rampa = 8000
tiempo_ch1ch1 = v_ch1ch1/rampa


#%%

num_samples = 125000
samplerate = 125000
#t = np.arange(0,num_samples/samplerate,1/samplerate)
t = np.linspace(0,(num_samples-1)/samplerate,num_samples)

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai2:3",max_val=10., min_val=-10.,name_to_assign_to_channel="ch1",terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)#, "Voltage")#,AIVoltageUnits.Volts)#,max_val=10., min_val=-10.)
    #task.ai_channels.add_ai_voltage_chan("Dev1/ai4")
    #task.ai_channels.ai_impedance  #(nidaqmx.constants.Impedance1.ONE_M_OHM)# =  # = nidaqmx.constants.Impedance1.ONE_M_OHM
    
    task.timing.cfg_samp_clk_timing(samplerate,samps_per_chan=num_samples)

    b=task.read(number_of_samples_per_channel=num_samples)


medicion = np.asarray(b)

#%%

carpeta_salida = 'Settingtime'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
    


np.save(os.path.join(carpeta_salida, 'medicion_cuadrada_200hz_125khz'),medicion)

medicion = np.transpose(medicion)
plt.plot(medicion[:,0],'.',label='CH2')
plt.plot(medicion[:,1],'.',label='CH3')
plt.legend()

v_ch1ch2 = 0.12
rampa = 8000
tiempo_ch1ch2 = v_ch1ch2/rampa

v_ch1ch1 = 0.160
rampa = 8000
tiempo_ch1ch1 = v_ch1ch1/rampa

