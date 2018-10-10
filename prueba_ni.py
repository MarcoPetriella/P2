import nidaqmx
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft

num_samples = 100000
samplerate = 100000
#t = np.arange(0,num_samples/samplerate,1/samplerate)
t = np.linspace(0,(num_samples-1)/samplerate,num_samples)
a=np.zeros([1,num_samples])
i=0
with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai0",max_val=5, min_val=-5)
    #task.ai_channels.add_ai_voltage_chan("Dev1/ai4")
    task.timing.cfg_samp_clk_timing(samplerate,samps_per_chan=num_samples)
    a[i,:]=task.read(number_of_samples_per_channel=num_samples)
    #print(a)

f = []
fft_a = abs(fft.fft(a,axis=1))
fft_a = fft_a[:,0:int(a.shape[1]/2)+1]
freq = np.linspace(0,int(samplerate/2),int(a.shape[1]/2+1))

plt.figure()
plt.plot(t,np.transpose(a),'.')
plt.figure()
plt.plot(freq,fft_a[0,:])

plt.show()
