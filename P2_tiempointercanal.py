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


#%%

import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import os
import matplotlib.pylab as pylab


params = {'legend.fontsize': 14,
          'figure.figsize': (14, 9),
         'axes.labelsize': 24,
         'axes.titlesize':18,
         'font.size':18,
         'xtick.labelsize':24,
         'ytick.labelsize':24}
pylab.rcParams.update(params)

#%%
######## Analisis tiempo intercanal

carpeta_salida = 'Tiempointercanal'
subcarpeta_salida = 'cuadrada'
samplerate = 50000

medicion = np.load(os.path.join(carpeta_salida,subcarpeta_salida, 'medicion.npy'))
plt.plot(medicion)

t = np.arange(0,medicion.shape[0])/samplerate

media = np.mean(medicion[33000:49000,:], axis = 0)
ruido = np.std(medicion[33000:49000,:], axis = 0)
print(media)
print(ruido)

bins = 5000
hist_frec_tot = np.zeros([bins,medicion.shape[1]])
hist_tension_tot = np.zeros([bins,medicion.shape[1]])

for i in range(medicion.shape[1]):
    [hist_frec,hist_tension] = np.histogram(medicion[33000:49000,1],range=(0.8,1.2),bins=bins)
    hist_tension = hist_tension[1:]
    hist_frec_tot[:,i] = hist_frec
    hist_tension_tot[:,i] = hist_tension
    
    ind = np.argmax(hist_frec)
    v_max = hist_tension[ind]
    
    ind_r = np.argmin(np.abs(hist_frec - hist_frec[ind]/2))
    v_ruido = np.abs(2*(hist_tension[ind] - hist_tension[ind_r]))    


fig = plt.figure(dpi=250)
ax = fig.add_axes([.12, .15, .35, .8])
for i in range(1):
    ax.plot(t,medicion[:,1],linewidth=2)
ax.set_xlim([0.70,0.8])
ax.set_ylim([0.97,1.010])
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Tensión [V]')


ax.grid(linestyle='--',linewidth=0.5)

    
ax1 = fig.add_axes([.60, .15, .35, .8])
for i in range(1):
    ax1.bar(hist_tension_tot[:,i],hist_frec_tot[:,i],align='center',width=0.0003,linewidth=0.1,edgecolor=(0.9,0.9,0.9))
ax1.set_xlim([0.97,1.010])
ax1.grid(linestyle='--',linewidth=0.5)
ax1.set_xlabel('Tensión [V]')
ax1.set_ylabel('Frecuencia')

figname = os.path.join(carpeta_salida, 'ruido.png')
fig.savefig(figname, dpi=300)  
plt.close(fig) 


##
from P2_funciones import fft_power_density

samplerate = 50000
medicion_fft = medicion[33000:49000,0]
t = np.arange(0,medicion_fft.shape[0])/samplerate

freq, psdx = fft_power_density(medicion_fft,samplerate)

plt.semilogy(freq,psdx)

plt.plot(t,medicion_fft)



##

fig = plt.figure(dpi=250)
ax = fig.add_axes([.15, .15, .75, .8])
for i in range(1):
    ax.plot(t,medicion[:,1],linewidth=2)
ax.set_xlim([0.70,0.8])
ax.set_ylim([0.97,1.010])
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Tensión [V]')
ax.grid(linestyle='--',linewidth=0.5)

figname = os.path.join(carpeta_salida, 'tension_sep.png')
fig.savefig(figname, dpi=300)  
plt.close(fig) 



fig = plt.figure(dpi=250) 
ax1 = fig.add_axes([.15, .15, .75, .8])
for i in range(1):
    ax1.plot(hist_tension_tot[:,i],hist_frec_tot[:,i],linewidth=2)
ax1.set_xlim([0.97,1.010])
ax1.grid(linestyle='--',linewidth=0.5)
ax1.set_xlabel('Tensión [V]')
ax1.set_ylabel('Frecuencia')

figname = os.path.join(carpeta_salida, 'ruido_sep.png')
fig.savefig(figname, dpi=300)  
plt.close(fig) 




#%%



def tiempo_intercanal(medicion,samplerate,vpp_por_dos,frec_rampa,carpeta_salida,subcarpeta_salida,fig_name):

    #t = np.arange(0,medicion.shape[0])/samplerate
    
    # Tiempo intercanal
    delta_t_rampa = 1/frec_rampa
    pendiente_dv_dt = vpp_por_dos/delta_t_rampa

    medicion0 = np.zeros([medicion.shape[0],medicion.shape[1]])
    for i in range(medicion.shape[1]):
        medicion0[:,i] = medicion[:,0]
    
    delta_v = medicion - medicion0
    delta_t = delta_v/pendiente_dv_dt
    
    bins = 3000
    frec_hist = np.zeros([bins,medicion.shape[1]-1])
    t_hist = np.zeros([bins,medicion.shape[1]-1])
    
    t_intecanales = np.zeros(medicion.shape[1]-1)
    ruido_t_intercanales = np.zeros(medicion.shape[1]-1)
    for i in range(medicion.shape[1]-1):
    
        [frec_i,t_i] = np.histogram(delta_t[:,i+1],range=(0,0.0001),bins=bins)
        frec_hist[:,i] = frec_i
        t_hist[:,i] = t_i[1:]
        
        ind = np.argmax(frec_i)
        t_intecanales[i] = t_i[ind]
        
        ind_r = np.argmin(np.abs(frec_i - frec_i[ind]/2))
        ruido_t_intercanales[i] = np.abs(2*(t_i[ind] - t_i[ind_r]))

    
    fig = plt.figure(dpi=250)
    ax = fig.add_axes([.15, .15, .75, .8])
    for i in range(t_hist.shape[1]):
        ax.plot(t_hist[:,i]*1e6,frec_hist[:,i],linewidth=2,label='Ch' + str(i+2) +' - Ch1')
    ax.axvline(1e6/samplerate,linestyle='--',color='red',label='Ch1 - Ch1')
    ax.grid(linestyle='--',linewidth=0.5)
    ax.set_xlim([0,1.25*1e6/samplerate])
    ax.set_xlabel('Tiempo intercanal [$\mu$s]')
    ax.set_ylabel('Frecuencia')
    ax.legend(loc=1)
    
    figname = os.path.join(carpeta_salida,subcarpeta_salida, 'intercanal'+fig_name+'.png')
    fig.savefig(figname, dpi=300)  
    plt.close(fig) 
    
    return pendiente_dv_dt,frec_hist,t_hist,t_intecanales,ruido_t_intercanales


carpeta_salida = 'Tiempointercanal'
subcarpeta_salida = 'rampa_2000hz_2ch'
samplerate = 50000
frec_rampa = 2000
vpp_por_dos = 2*2

medicion0 = np.load(os.path.join(carpeta_salida,subcarpeta_salida, 'medicion.npy'))
medicion0 = np.transpose(medicion0)

medicion = np.zeros([medicion0.shape[0]-1,medicion0.shape[1]])
medicion[:,0] = medicion0[0:-1,0]
medicion[:,1] = medicion0[1:,0]
t = np.arange(0,medicion.shape[0])/samplerate


pendiente_dv_dt,frec_hist,t_hist,t_intecanales0,ruido_t_intercanales0 = tiempo_intercanal(medicion,samplerate,vpp_por_dos,frec_rampa,carpeta_salida,subcarpeta_salida,'0')
t_medio_intercanal0 = t_intecanales0[0]
error_intercanal0 = ruido_t_intercanales0[0]

###

carpeta_salida = 'Tiempointercanal'
subcarpeta_salida = 'rampa_2000hz_2ch'
samplerate = 50000
frec_rampa = 2000
vpp_por_dos = 2*2

medicion = np.load(os.path.join(carpeta_salida,subcarpeta_salida, 'medicion.npy'))
medicion = np.transpose(medicion)
t = np.arange(0,medicion.shape[0])/samplerate

pendiente_dv_dt,frec_hist,t_hist,t_intecanales1,ruido_t_intercanales1 = tiempo_intercanal(medicion,samplerate,vpp_por_dos,frec_rampa,carpeta_salida,subcarpeta_salida,'2')
t_medio_intercanal1 = t_intecanales1[0]
error_intercanal1 = ruido_t_intercanales1[0]

##

carpeta_salida = 'Tiempointercanal'
subcarpeta_salida = 'rampa_2000hz'
samplerate = 50000
frec_rampa = 2000
vpp_por_dos = 2*2

medicion = np.load(os.path.join(carpeta_salida,subcarpeta_salida, 'medicion.npy'))
medicion = np.transpose(medicion)
t = np.arange(0,medicion.shape[0])/samplerate

pendiente_dv_dt,frec_hist,t_hist,t_intecanales2,ruido_t_intercanales2 = tiempo_intercanal(medicion,samplerate,vpp_por_dos,frec_rampa,carpeta_salida,subcarpeta_salida,'4')
t_medio_intercanal2 = np.mean(np.diff(t_intecanales2))
error_intercanal2 = ruido_t_intercanales2[0]

##

carpeta_salida = 'Tiempointercanal'
subcarpeta_salida = 'rampa_2000hz_5ch'
samplerate = 50000
frec_rampa = 2000
vpp_por_dos = 2*2

medicion = np.load(os.path.join(carpeta_salida,subcarpeta_salida, 'medicion.npy'))
medicion = np.transpose(medicion)
t = np.arange(0,medicion.shape[0])/samplerate

pendiente_dv_dt,frec_hist,t_hist,t_intecanales3,ruido_t_intercanales3 = tiempo_intercanal(medicion,samplerate,vpp_por_dos,frec_rampa,carpeta_salida,subcarpeta_salida,'5')
t_medio_intercanal3 = np.mean(np.diff(t_intecanales3))
error_intercanal3 = ruido_t_intercanales3[0]


####
carpeta_salida = 'Tiempointercanal'
subcarpeta_salida = 'rampa_2000hz_5ch'

fig = plt.figure(dpi=250)
ax = fig.add_axes([.15, .15, .75, .8])
for i in range(medicion.shape[1]):
    ax.plot(medicion[:,i],'.',markersize=5,label='CH' + str(i+1))
ax.set_xlim([-5,35])
ax.set_xlabel('Muestra')
ax.set_ylabel('Tensión [V]')
ax.grid(linestyle='--',linewidth=0.5)
ax.legend()
figname = os.path.join(carpeta_salida, subcarpeta_salida,'intercanal_tension.png')
fig.savefig(figname, dpi=300)  
plt.close(fig) 


####
canales = np.array([1,2,4,5])
t_inter = np.array([t_medio_intercanal0,t_medio_intercanal1,t_medio_intercanal2,t_medio_intercanal3])
error_t_inter = np.array([0,error_intercanal1,error_intercanal2,error_intercanal3])

t_teo = 1/(np.arange(1,6))
canales_teo = np.arange(1,6)


fig = plt.figure(dpi=250)
ax = fig.add_axes([.15, .15, .75, .8])
#ax.errorbar(canales,samplerate*t_inter,yerr=samplerate*error_t_inter,fmt='o')
ax.plot(canales,samplerate*t_inter,'o',markersize=10,label = 'Medido')
ax.plot(canales_teo,t_teo,linewidth=2,label = 'Esperado')
ax.set_xlabel('Nro. canales')
ax.set_ylabel('Tiempo Intercanal normalizado') #[$\Delta$t * frec_sampleo]
ax.legend()
ax.grid(linestyle='--',linewidth=0.5)
ax.set_ylim([0,1.2])
ax.set_xticks([1,2,3,4,5])

figname = os.path.join(carpeta_salida, 'intercanal_tot.png')
fig.savefig(figname, dpi=300)  
plt.close(fig) 


## ruido campana

ruido_en_t = v_ruido/pendiente_dv_dt