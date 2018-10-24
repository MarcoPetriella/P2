import nidaqmx
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import os

num_samples = 100000
samplerate = 50000
#t = np.arange(0,num_samples/samplerate,1/samplerate)
t = np.linspace(0,(num_samples-1)/samplerate,num_samples)
a=np.zeros([1,num_samples*2])
i=0

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai3:5",max_val=10., min_val=-10.,name_to_assign_to_channel="ch1",terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)#, "Voltage")#,AIVoltageUnits.Volts)#,max_val=10., min_val=-10.)
    #task.ai_channels.add_ai_voltage_chan("Dev1/ai4")
    #task.ai_channels.ai_impedance  #(nidaqmx.constants.Impedance1.ONE_M_OHM)# =  # = nidaqmx.constants.Impedance1.ONE_M_OHM
    
    task.timing.cfg_samp_clk_timing(samplerate,samps_per_chan=num_samples)

    b=task.read(number_of_samples_per_channel=num_samples)
    #print(a)


carpeta_salida = 'USB6210'
subcarpeta_salida = 'RSE'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
    
if not os.path.exists(os.path.join(carpeta_salida,subcarpeta_salida)):
    os.mkdir(os.path.join(carpeta_salida,subcarpeta_salida))     

np.save(os.path.join(carpeta_salida,subcarpeta_salida, 'a'),a)

f = []
fft_a = abs(fft.fft(a,axis=1))
fft_a = fft_a[:,0:int(a.shape[1]/2)+1]
freq = np.linspace(0,int(samplerate/2),int(a.shape[1]/2+1))

plt.figure()
plt.plot(t,np.transpose(a),'.')
#plt.plot(t,a[0,:])
plt.figure()
plt.plot(freq,fft_a[0,:])

plt.show()


#######################
#%%


import nidaqmx
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import os

num_samples = 100000
samplerate = 100000
#t = np.arange(0,num_samples/samplerate,1/samplerate)
i=0
frecs = [1,10,30,45,49,50,51,55,60,70,80,90,95,100,105,120,130,140,145,149,150,151]
a=np.zeros([len(frecs),num_samples])

#%%
with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai4",max_val=10., min_val=-10.,name_to_assign_to_channel="ch1",terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)#, "Voltage")#,AIVoltageUnits.Volts)#,max_val=10., min_val=-10.)
    #task.ai_channels.add_ai_voltage_chan("Dev1/ai4")
    #task.ai_channels.ai_impedance  #(nidaqmx.constants.Impedance1.ONE_M_OHM)# =  # = nidaqmx.constants.Impedance1.ONE_M_OHM
    
    task.timing.cfg_samp_clk_timing(samplerate,samps_per_chan=num_samples)
    a[i,:]=task.read(number_of_samples_per_channel=num_samples)
    #print(a)


f = []
fft_a = abs(fft.fft(a,axis=1))
fft_a = fft_a[:,0:int(a.shape[1]/2)+1]
freq = np.linspace(0,int(samplerate/2),int(a.shape[1]/2+1))
t = np.linspace(0,(num_samples-1)/samplerate,num_samples)


plt.figure()
plt.plot(t,a[i,:],'-')
#plt.plot(t,a[0,:])
plt.figure()
plt.plot(freq,fft_a[i,:])

plt.show()

i = i + 1

#%%
carpeta_salida = 'nyquist'
subcarpeta_salida = 'frec_vs_frec'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
    
if not os.path.exists(os.path.join(carpeta_salida,subcarpeta_salida)):
    os.mkdir(os.path.join(carpeta_salida,subcarpeta_salida))     

np.save(os.path.join(carpeta_salida,subcarpeta_salida, 'a'),a)
np.save(os.path.join(carpeta_salida,subcarpeta_salida, 'frecs'),frecs)


