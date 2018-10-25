import nidaqmx
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import os
from P2_funciones import cross_correlation_using_fft
import nidaqmx.constants as constants

#%%
num_samples = 5000
samplerate = 50000
#t = np.arange(0,num_samples/samplerate,1/samplerate)
t = np.linspace(0,(num_samples-1)/samplerate,num_samples)

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai2",max_val=2., min_val=-2.,name_to_assign_to_channel="ch1",terminal_config=constants.TerminalConfiguration.RSE)#, "Voltage")#,AIVoltageUnits.Volts)#,max_val=10., min_val=-10.)
    #task.ai_channels.add_ai_voltage_chan("Dev1/ai4")
    #task.ai_channels.ai_impedance  #(nidaqmx.constants.Impedance1.ONE_M_OHM)# =  # = nidaqmx.constants.Impedance1.ONE_M_OHM
    
    task.timing.cfg_samp_clk_timing(samplerate,samps_per_chan=num_samples,sample_mode=constants.AcquisitionType.CONTINUOUS)


    while True:
        b=task.read(number_of_samples_per_channel=num_samples)
        medicion = np.asarray(b)
        
  
#%%
      
with nidaqmx.Task() as task:
    task.co_channels.add_co_pulse_chan_freq(counter='Dev1/ctr0',duty_cycle=0.5,freq=20.0)

    pulse_count = 1000
    
    task.timing.cfg_implicit_timing(sample_mode=constants.AcquisitionType.FINITE,samps_per_chan=pulse_count)
    task.triggers.start_trigger.cfg_dig_edge_start_trig('Dev1/ctr0')
    task.triggers.start_trigger.retriggerable=True

    task.start()
    


#%%