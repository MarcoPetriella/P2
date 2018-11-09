
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
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, CheckButtons
import os

class pid_daq(object):

    def __init__(self, parametros):       
        self.parametros = parametros

    def warning_callback(self,warning_string): 
        self.evento_warning.set()
        self.warnings.append(warning_string)  

    def exit_callback1(self,error_string):   
        self.acquiring_errors.append(error_string) 
        self.evento_salida.set() 

    def print_error(self,string_list):       
        s_tot = ''
        for i in range(len(string_list)):
            s_tot = s_tot + string_list[i] + '\n' 
        self.txt_error.set_text(s_tot)    
        
    def inicializa_errores(self):
        self.initialize_errors = []
        self.acquiring_errors = []  
        self.warnings = []
        self.evento_warning = threading.Event()
        self.evento_salida = threading.Event()
        
    def inicializa_variables(self):
        self.buffer_chunks = parametros['buffer_chunks']   
        self.ai_samples = parametros['ai_samples']
        self.ai_samplerate = parametros['ai_samplerate']
        self.ai_device = parametros['ai_device']
        self.ai_channels = parametros['ai_channels']
        self.do_device = parametros['do_device']
        self.do_channel = parametros['do_channel']
        self.initial_do_duty_cycle = parametros['initial_do_duty_cycle']
        self.initial_do_frequency = parametros['initial_do_frequency']        
        self.setpoint = parametros['setpoint']
        self.save_raw_data = parametros['save_raw_data']
        self.save_processed_data = parametros['save_processed_data']
        self.show_plot = parametros['show_plot']
        self.path_data_save = parametros['path_data_save']      
        self.callback_pid = parametros['callback_pid']
        self.callback_pid_variables = parametros['callback_pid_variables']           
        parametros_pid = parametros['pid_constants']
        self.kp = parametros_pid[0]
        self.ki = parametros_pid[1]
        self.kd = parametros_pid[2]
        self.isteps = parametros_pid[3]  
        
        # Valores predeterminado
        self.pid_onoff_button = [True]
        self.plot_fontsize = 10
        self.default_fontsize = 6

        self.setpoint_min = 0
        self.setpoint_max = 5
        self.kp_min = 0
        self.kp_max = 10
        self.ki_min = 0
        self.ki_max = 60
        self.kd_min = 0
        self.kd_max = 60
        
        self.ax_lim_i = 0
        self.ax_lim_f = 5
        self.ax1_lim_i = 0
        self.ax1_lim_f = 1.2
        self.ax2_lim_i = -2
        self.ax2_lim_f = 2      
        
        self.sample_period = self.ai_samples/self.ai_samplerate
               
        
    def acondiciona_variables(self):
        # Archivos de salida
        self.path_raw_data = self.path_data_save + '_raw_data.bin'
        self.path_duty_cycle_data = self.path_data_save + '_duty_cycle.bin'
        self.path_mean_data = self.path_data_save + '_mean_data.bin'
        self.path_pid_constants = self.path_data_save + '_pid_constants.bin'
        self.path_pid_terminos = self.path_data_save + '_pid_terminos.bin'
        
        if self.save_raw_data:
            if os.path.exists(self.path_raw_data):
                os.remove(self.path_raw_data)
                
        if self.save_processed_data:
            if os.path.exists(self.path_duty_cycle_data):
                os.remove(self.path_duty_cycle_data)
        
            if os.path.exists(self.path_mean_data):
                os.remove(self.path_mean_data)        
    
            if os.path.exists(self.path_pid_constants):
                os.remove(self.path_pid_constants)    
    
            if os.path.exists(self.path_pid_terminos):
                os.remove(self.path_pid_terminos)           
        
        parametros = self.parametros
        buffer_chunks = self.buffer_chunks
        ai_samplerate = self.ai_samplerate
        ai_samples = self.ai_samples
        initial_do_frequency = self.initial_do_frequency
        ai_device = self.ai_device
        ai_channels = self.ai_channels
        do_device = self.do_device
        do_channel = self.do_channel
        sub_chunk_save = np.NaN
        sub_chunk_plot = np.NaN
        initialize_errors = []
        warnings = []
        
        # Errores de inicializacion
        if 'sub_chunk_save' in parametros:
            sub_chunk_save = parametros['sub_chunk_save'] 
            if buffer_chunks%sub_chunk_save != 0:
                initialize_errors.append('buffer_chunks debe ser multiplo de sub_chunk_save')                
        else:
            i = 20
            while i > 1:
                sub_chunk_save = buffer_chunks/i
                if buffer_chunks%i == 0:
                    break
                i = i - 1
            if i == 1:
                initialize_errors.append('No se encuentra sub_chunk_save tal que buffer_chunks sea multiplo')   
        sub_chunk_save = int(sub_chunk_save)  
            
        if 'sub_chunk_plot' in parametros:    
            sub_chunk_plot = parametros['sub_chunk_plot'] 
            if buffer_chunks%sub_chunk_plot != 0:
                initialize_errors.append('buffer_chunks debe ser multiplo de sub_chunk_plot')    
        else:
            i = 20
            while i > 1:
                sub_chunk_plot = buffer_chunks/i
                if buffer_chunks%i == 0:
                    break
                i = i - 1
            if i == 1:
                initialize_errors.append('No se encuentra sub_chunk_plot tal que buffer_chunks sea multiplo')     
        
        if 'plot_rate_hz' in parametros:
            plot_rate_hz = parametros['plot_rate_hz']
            sub_chunk_plot = ai_samplerate/ai_samples/plot_rate_hz
            if buffer_chunks%sub_chunk_plot != 0 or sub_chunk_plot != int(sub_chunk_plot):
                warnings.append('El plot_rate_hz indicado no es posible, se busca el más cercano')
                sub_chunk_plot = int(sub_chunk_plot*1.5)
                while sub_chunk_plot > 1:
                    if buffer_chunks%sub_chunk_plot == 0:
                        break
                    sub_chunk_plot = int(sub_chunk_plot-1)
                if sub_chunk_plot == 1:
                    initialize_errors.append('No se encuentra sub_chunk_plot tal que buffer_chunks sea multiplo')  
        sub_chunk_plot = int(sub_chunk_plot)
        
        if (ai_samples/ai_samplerate)%(1/initial_do_frequency) != 0.:
            warnings.append(warning_string = 'La cantidad de ciclos del PWM no es entera!')
    
        # pasos de la integral del pid
        paso_integral = 10
        possible_isteps = np.arange(0,buffer_chunks+paso_integral,paso_integral,dtype=int)
        possible_isteps[0] = 1
    
        # Largo del vector a graficar
        if 'nbr_buffers_plot' in parametros:
            nbr_buffers_plot = int(parametros['nbr_buffers_plot'])
        else:
            nbr_buffers_plot = 10
            
        # ai string
        ai_channels_str = ''
        ai_nbr_channels = 0
        if len(ai_channels) == 2:
            ai_channels_str = str(ai_channels[0]) + ':' + str(ai_channels[1])
            ai_nbr_channels = ai_channels[1] - ai_channels[0] + 1
        elif len(ai_channels) == 1:
            ai_channels_str = str(ai_channels[0])
            ai_nbr_channels = 1
        else:
            initialize_errors.append('El formato de los canales no está bien especificado')  
            
        ai_channels_str = ai_device + ai_channels_str
    
        # do string
        do_channels_str = do_device + str(do_channel[0])    
        if len(do_channel) > 1:
            initialize_errors.append('Por ahora solo está habilitado un solo canal digital')   
       
        
        self.sub_chunk_save = sub_chunk_save
        self.sub_chunk_plot = sub_chunk_plot
        self.possible_isteps = possible_isteps
        self.nbr_buffers_plot = nbr_buffers_plot
        self.ai_channels_str = ai_channels_str
        self.ai_nbr_channels = ai_nbr_channels
        self.do_channels_str = do_channels_str
        self.initialize_errors = initialize_errors
        self.warnings = warnings

    def define_limites_de_plots(self,
        ax_lim_i,
        ax_lim_f,
        ax1_lim_i,
        ax1_lim_f,
        ax2_lim_i,
        ax2_lim_f
    ):
        
        self.ax_lim_i = ax_lim_i
        self.ax_lim_f = ax_lim_f
        self.ax1_lim_i = ax1_lim_i
        self.ax1_lim_f = ax1_lim_f
        self.ax2_lim_i = ax2_lim_i
        self.ax2_lim_f = ax2_lim_f      

    def define_limites_sliders(self,
        setpoint_min,
        setpoint_max,
        kp_min,
        kp_max,
        ki_min,
        ki_max,
        kd_min,
        kd_max
    ):
        
        self.setpoint_min = setpoint_min
        self.setpoint_max = setpoint_max
        self.kp_min = kp_min
        self.kp_min = kp_min
        self.ki_min = ki_min
        self.ki_min = ki_min  
        self.kd_min = kd_min
        self.kd_min = kd_min 
        
    def inicializa_interfaz(self):

        default_fontsize = self.default_fontsize
        plot_fontsize = self.plot_fontsize
        ax_lim_i = self.ax_lim_i
        ax_lim_f = self.ax_lim_f
        ax1_lim_i = self.ax1_lim_i
        ax1_lim_f = self.ax1_lim_f
        ax2_lim_i = self.ax2_lim_i
        ax2_lim_f = self.ax2_lim_f        
        
        
        buffer_chunks = self.buffer_chunks
        nbr_buffers_plot = self.nbr_buffers_plot
        ai_nbr_channels = self.ai_nbr_channels
        ai_samples = self.ai_samples
        ai_samplerate = self.ai_samplerate
        setpoint = self.setpoint
        kp =self.kp
        ki = self.ki
        kd = self.kd
        isteps = self.isteps        
        initial_do_frequency = self.initial_do_frequency
        sub_chunk_plot = self.sub_chunk_plot
        save_raw_data = self.save_raw_data
        save_processed_data = self.save_processed_data
        
        data_plot1 = np.zeros([buffer_chunks*nbr_buffers_plot,ai_nbr_channels])
        data_plot2 = np.zeros(buffer_chunks*nbr_buffers_plot)    
        data_plot3 = np.zeros([buffer_chunks*nbr_buffers_plot,3])  
        
        tiempo_vec = np.arange(0,data_plot1.shape[0])/data_plot1.shape[0]
        tiempo_vec = tiempo_vec*(ai_samples*buffer_chunks*nbr_buffers_plot/ai_samplerate)
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")
      
        fig = plt.figure(figsize=(7,3.7),dpi=250)
        ax = fig.add_axes([.08, .35, .70, .33])  
        ax1 = ax.twinx()
        
        line1 = []
        for i in range(ai_nbr_channels):
            line, = ax.plot(tiempo_vec,data_plot1[:,i], '-')  
            line1.append(line)
        line2, = ax1.plot(tiempo_vec,data_plot2, '-',color='red')
        txt_now = ax.text(1.01,1.1,now,fontsize=default_fontsize,transform = ax.transAxes)
        ax.set_ylim([ax_lim_i,ax_lim_f])
        ax1.set_ylim([ax1_lim_i,ax1_lim_f])
        ax.set_xlabel('tiempo [s]',fontsize = plot_fontsize)
        ax1.set_ylabel('duty cycle',fontsize = plot_fontsize)
        ax.set_ylabel('mean [V]',fontsize = plot_fontsize)   
        setpoint_line = ax.axhline(setpoint,linestyle='--',linewidth=0.8)
        
        ax.xaxis.set_tick_params(labelsize=plot_fontsize)
        ax1.xaxis.set_tick_params(labelsize=plot_fontsize)
        ax.yaxis.set_tick_params(labelsize=plot_fontsize)
        ax1.yaxis.set_tick_params(labelsize=plot_fontsize)   
        ax.grid(linestyle='--',linewidth=0.3)   
    
        ##
        ax3 = fig.add_axes([.08, .71, .70, .27])  
        ax3.set_xticklabels([])
        line3 = []
        for i in range(data_plot3.shape[1]):
            line, = ax3.plot(tiempo_vec,data_plot3[:,i], '-')  
            line3.append(line) 
        ax3.set_ylim([ax2_lim_i,ax2_lim_f])
        ax3.legend(['P','I','D'],bbox_to_anchor=(1.01, 1.0))
        
        ax3.xaxis.set_tick_params(labelsize=plot_fontsize)
        ax3.yaxis.set_tick_params(labelsize=plot_fontsize) 
        ax3.grid(linestyle='--',linewidth=0.3)    
        
        ##         
        ax2 = fig.add_axes([.15, .03, .3, .3])        
        ax2.axis('off')
        xi = 0.68
        yi = 0.65
        dyi = 0.10
        ax2.text(xi,yi,'Pending processes',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
        ax2.text(xi,yi - 1*dyi,'Input buffer filling: ',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
        ax2.text(xi,yi - 2*dyi,'Output buffer emptying:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
        ax2.text(xi,yi - 3*dyi,'Raw data writer:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
        ax2.text(xi,yi - 4*dyi,'Processed data writer:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
        ax2.text(xi,yi - 5*dyi,'Plotting:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
        ax2.text(xi,yi - 6*dyi,'Measuring acquiring ratio:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
        
        xi = 1.53
        monitoring_txt0 = ax2.text(xi,yi - 1*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
        monitoring_txt1 = ax2.text(xi,yi - 2*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
        monitoring_txt2 = ax2.text(xi,yi - 3*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
        monitoring_txt3 = ax2.text(xi,yi - 4*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
        monitoring_txt4 = ax2.text(xi,yi - 5*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
        monitoring_txt5 = ax2.text(xi,yi - 6*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
        
        xi = -0.40
        ax2.text(xi,yi,'Acquisition parameters',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
        ax2.text(xi,yi - 1*dyi,'Samplerate: ',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
        ax2.text(xi,yi - 2*dyi,'Samples per chunk:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
        ax2.text(xi,yi - 3*dyi,'Nbr. chunks:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
        ax2.text(xi,yi - 4*dyi,'PWM frequency:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
        ax2.text(xi,yi - 5*dyi,'Nbr. PWM cycles per chunk:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
        ax2.text(xi,yi - 6*dyi,'Save raw / processed data:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
        ax2.text(xi,yi - 7*dyi,'Display plot chunks / rate:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
        
        xi = 0.55
        ax2.text(xi,yi - 1*dyi,'%6.2f' % (ai_samplerate/1000.0) + ' kHz',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
        ax2.text(xi,yi - 2*dyi,'%4d' % ai_samples + ' / ' + '%6.2f' % (ai_samples/ai_samplerate*1000.) + ' ms',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
        ax2.text(xi,yi - 3*dyi,'%4d' % buffer_chunks + ' / ' + '%6.2f' % (buffer_chunks*ai_samples/ai_samplerate) + ' s',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
        ax2.text(xi,yi - 4*dyi,'%6.2f' % (initial_do_frequency/1000.) + ' kHz',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
        ax2.text(xi,yi - 5*dyi,'%6.2f' % ((ai_samples/ai_samplerate)*(initial_do_frequency)) ,fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
        ax2.text(xi,yi - 6*dyi, str(save_raw_data) + ' / ' + str(save_processed_data),fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
        ax2.text(xi,yi - 7*dyi, '%4d' % sub_chunk_plot + ' / ' +'%6.2f' % (ai_samplerate/ai_samples/sub_chunk_plot) + ' Hz',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
    
        xi = 2.35
        txt_pid_parameters = []
        ax2.text(xi,yi,'PID parameters',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
        pid_para_txt0 = ax2.text(xi,yi - 1*dyi,'%6.2f' % (setpoint) + ' V',fontsize=default_fontsize,va='center',ha='left',transform = ax2.transAxes)
        pid_para_txt1 = ax2.text(xi,yi - 2*dyi,'%6.2f' % (kp),fontsize=default_fontsize,va='center',ha='left',transform = ax2.transAxes)
        pid_para_txt2 = ax2.text(xi,yi - 3*dyi,'%6.2f' % (ki),fontsize=default_fontsize,va='center',ha='left',transform = ax2.transAxes)
        pid_para_txt3 = ax2.text(xi,yi - 4*dyi,'%6.2f' % (kd),fontsize=default_fontsize,va='center',ha='left',transform = ax2.transAxes)
        pid_para_txt4 = ax2.text(xi+0.03,yi - 5*dyi,'%2d' % (isteps),fontsize=default_fontsize,va='center',ha='left',transform = ax2.transAxes)

    
        xi = 1.63
        ax2.text(xi,yi,'PID parameters',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
        
        # Mensje de error
        x_error = 1.64
        y_error = 0.01
    
        txt_error = ax2.text(x_error,y_error,'',fontsize=default_fontsize-1,va='center',transform = ax2.transAxes,color='red') 

        self.data_plot1 = data_plot1
        self.data_plot2 = data_plot2
        self.data_plot3 = data_plot3
        self.fig = fig
        self.ax = ax
        self.ax1 = ax1
        self.ax2 = ax2
        self.line1 = line1
        self.line2 = line2
        self.line3 = line3
        self.setpoint_line = setpoint_line
        self.txt_now = txt_now
        self.txt_pid_parameters = txt_pid_parameters
        self.monitoring_txt0 = monitoring_txt0
        self.monitoring_txt1 = monitoring_txt1
        self.monitoring_txt2 = monitoring_txt2
        self.monitoring_txt3 = monitoring_txt3
        self.monitoring_txt4 = monitoring_txt4
        self.monitoring_txt5 = monitoring_txt5
        self.pid_para_txt0 = pid_para_txt0
        self.pid_para_txt1 = pid_para_txt1
        self.pid_para_txt2 = pid_para_txt2
        self.pid_para_txt3 = pid_para_txt3
        self.pid_para_txt4 = pid_para_txt4
        self.txt_error = txt_error

        ## Botones y sliders        
    def inicializa_sliders_botones(self):
        
        default_fontsize = self.default_fontsize
        setpoint_min = self.setpoint_min
        setpoint_max = self.setpoint_max
        kp_min = self.kp_min
        kp_max = self.kp_max
        ki_min = self.ki_min
        ki_max = self.ki_max
        kd_min = self.kd_min
        kd_max = self.kd_max
        
        setpoint = self.setpoint
        kp = self.kp
        ki = self.ki
        kd = self.kd
        isteps = self.isteps
        
        xi_s = 0.69
        yi_s = 0.185
        dyi_s = 0.03        
        slider_width = 0.12
        slider_height = 0.02        
                  
        axstop = plt.axes([0.88, 0.35, 0.1, 0.075])
        bstop = Button(axstop, 'Stop')
        bstop.on_clicked(self.exit_callback) 
        
        axonoff = plt.axes([0.88, 0.25, 0.1, 0.075])
        bonoff = Button(axonoff, 'PID ON ')
        bonoff.on_clicked(self.pid_onoff)  
        bonoff.label.set_size(10)
        bonoff.label.set_backgroundcolor([1,1,1,0.8])
        bonoff.label.set_color([0.,0.99,0.])
        
        # Slider setpoit
        axcolor = 'lightgoldenrodyellow'
        axsetpoint = plt.axes([xi_s, yi_s, slider_width, slider_height], facecolor=axcolor)
        ssetpoint = Slider(axsetpoint, 'Setpoint',setpoint_min, setpoint_max, valinit=setpoint)
        ssetpoint.on_changed(self.update_pid) 
        ssetpoint.label.set_size(default_fontsize)
        ssetpoint.valtext.set_size(default_fontsize)
    
        # Slider kp
        axkp = plt.axes([xi_s, yi_s - 1*dyi_s, slider_width, slider_height], facecolor=axcolor)
        skp = Slider(axkp, 'kp',kp_min, kp_max, valinit=kp)
        skp.on_changed(self.update_pid) 
        skp.label.set_size(default_fontsize)
        skp.valtext.set_size(default_fontsize)
    
        # Slider ki
        axki = plt.axes([xi_s, yi_s - 2*dyi_s, slider_width, slider_height], facecolor=axcolor)
        ski = Slider(axki, 'ki',ki_min, ki_max, valinit=ki)
        ski.on_changed(self.update_pid) 
        ski.label.set_size(default_fontsize)
        ski.valtext.set_size(default_fontsize)
        
        # Slider kd
        axkd = plt.axes([xi_s, yi_s - 3*dyi_s, slider_width, slider_height], facecolor=axcolor)
        skd = Slider(axkd, 'kd',kd_min, kd_max, valinit=kd)
        skd.on_changed(self.update_pid)   
        skd.label.set_size(default_fontsize)
        skd.valtext.set_size(default_fontsize)        
    
        # Slider integral steps
        axisteps = plt.axes([xi_s, yi_s - 4*dyi_s, slider_width, slider_height], facecolor=axcolor)
        sisteps = Slider(axisteps, 'steps',0, buffer_chunks, valinit=isteps)
        sisteps.valfmt = '%2d'
        sisteps.on_changed(self.update_pid) 
        sisteps.label.set_size(default_fontsize)
        sisteps.valtext.set_size(default_fontsize)
        
        
        self.axstop = axstop
        self.bstop = bstop
        self.axonoff = axonoff
        self.bonoff = bonoff
        self.axsetpoint = axsetpoint
        self.ssetpoint = ssetpoint
        self.axkp = axkp
        self.skp = skp
        self.axki = axki
        self.ski = ski
        self.axkd = axkd
        self.skd = skd
        self.axisteps = axisteps
        self.sisteps = sisteps

                
    # Callback de botones
    def update_pid(self,val):
        
        possible_isteps = self.possible_isteps
                
        setpoint = self.ssetpoint.val
        kp = self.skp.val
        ki = self.ski.val
        kd = self.skd.val
        isteps = self.sisteps.val    
        ind_is = np.argmin(np.abs(possible_isteps - self.sisteps.val))       
        isteps = possible_isteps[ind_is]
        while self.sisteps.val != possible_isteps[ind_is]:
            self.sisteps.set_val(isteps)
            
        self.setpoint = setpoint
        self.kp = kp
        self.ki = ki
        self.lkd = kd
        self.isteps = isteps
        
        
    def pid_onoff(self,event):
        
        pid_onoff_button = self.pid_onoff_button
        bonoff = self.bonoff
        
        if pid_onoff_button is True:
            pid_onoff_button = False
            bonoff.label.set_text('PID OFF')
            bonoff.label.set_color([0.99,0.0,0.])
        else:
            pid_onoff_button = True  
            bonoff.label.set_text('PID ON ')
            bonoff.label.set_color([0.,0.99,0.])            
            
    def exit_callback(self,event):  
        self.acquiring_errors.append('Medición interrumpida por el usuario')
        self.evento_salida.set()
        

    def inicializa_buffers(self):
    
        buffer_chunks = self.buffer_chunks
        ai_samples = self.ai_samples
        ai_nbr_channels = self.ai_nbr_channels
        initial_do_duty_cycle = self.initial_do_duty_cycle
        setpoint = self.setpoint
        kp = self.kp
        ki = self.ki
        kd = self.kd
        isteps = self.isteps  
        
        input_buffer = np.zeros([buffer_chunks,ai_samples,ai_nbr_channels])
        output_buffer_mean_data = np.zeros([buffer_chunks,ai_nbr_channels])
        output_buffer_duty_cycle = np.ones(buffer_chunks)*initial_do_duty_cycle
        output_buffer_error_data = np.zeros(buffer_chunks)
        output_buffer_pid_constants = np.ones([buffer_chunks,5]) 
        output_buffer_pid_constants[:,0] = output_buffer_pid_constants[:,0]*setpoint
        output_buffer_pid_constants[:,1] = output_buffer_pid_constants[:,1]*kp
        output_buffer_pid_constants[:,2] = output_buffer_pid_constants[:,2]*ki
        output_buffer_pid_constants[:,3] = output_buffer_pid_constants[:,3]*kd
        output_buffer_pid_constants[:,4] = output_buffer_pid_constants[:,4]*isteps
        output_buffer_pid_terminos = np.zeros([buffer_chunks,3])  
        
        self.input_buffer = input_buffer
        self.output_buffer_mean_data = output_buffer_mean_data
        self.output_buffer_duty_cycle = output_buffer_duty_cycle
        self.output_buffer_error_data = output_buffer_error_data
        self.output_buffer_pid_constants = output_buffer_pid_constants
        self.output_buffer_pid_terminos = output_buffer_pid_terminos





       
#    # Semaforos
#    semaphore1 = threading.Semaphore(0) # Input buffer
#    semaphore2 = threading.Semaphore(0) # Output buffer
#    semaphore3 = threading.Semaphore(0) # Guardado de raw data
#    semaphore4 = threading.Semaphore(0) # Guardado de processed data
#    semaphore5 = threading.Semaphore(0) # Plot
    
    


    
    def configure_acquisition(self):
        self.inicializa_errores()
        self.inicializa_variables()
        self.acondiciona_variables()
        self.inicializa_buffers()        
        self.inicializa_interfaz()    
        self.inicializa_sliders_botones()
        if len(self.warnings) > 0:          
            self.print_error(self.warnings)
            
        if len(self.initialize_errors) > 0:
            self.initialize_errors = [u'Error de inicialización'] + self.initialize_errors
            self.print_error(self.initialize_errors)
            return
            
        

#%%
        
def callback_pid(i, input_buffer, output_buffer_duty_cycle, output_buffer_pid_terminos, output_buffer_mean_data, output_buffer_error_data, output_buffer_pid_constants, buffer_chunks, sample_period ,callback_pid_variables):

    """
    # Variables de entrada fijas:
    # --------------------------
    # i: posicion actual en el input buffer. Recordar que i [0,buffer_chunks]
    # input_buffer: np.array [buffer_chunks, samples, ai_nbr_channels]
    # output_buffer_duty_cycle: np.array [buffer_chunks]. La i-posicion es la vieja, y se la calcula en el callback (output_buffer_duty_cycle_i)
    # output_buffer_pid_terminos: np.array [buffer_chunks,3] [buffer_chunks,termino_p termino_i termino_d]. La i-posicion es la vieja, y se la calcula en el callback
    #   termino_p: termino multiplicativo
    #   termino_i: termino integral
    #   termino_d: termino derivativo
    # output_buffer_mean_data: np.array [buffer_chunks,ai_nbr_channels]. La i-posicion es la actual. La columna 0 se utiliza para el PID.
    # output_buffer_error_data: np.array [buffer_chunks]. La i-posicion es la actual: output_buffer_mean_data[i,0] - lsetpoint[0]. Importante: se toma con el valor del setpoint del slider!
    # buffer_chunks: Cantidad de chunks del buffer
    # output_buffer_pid_constants: np.array [buffer_chunks, 5] - > [buffer_chunks, setpoint kp ki kd isteps]. La i-posicion es la actual. Importante: se toma con el valor del setpoint del slider!
    #   setpoint: Valor de tensión del setpoint
    #   kp: constante multiplicativa del PID
    #   ki: constante integral del PID
    #   kd: constante derivativa del PID
    #   isteps: cantidad de pasos para atrás utilizados para el termino integral
    #
    #
    # Variables de entrada del usuario:
    # --------------------------------
    # callback_pid_variables: lista con variables puestas por el usuario
    #   Ejemplo:
    #   variable0 = callback_pid_variables[0]
    #   variable1 = callback_pid_variables[1]
    #
    #
    # Salidas (calculadas en el callback):
    # -------
    # output_buffer_duty_cycle_i: duty cycle calculado en el callback
    # output_buffer_error_data_i: error calculado en el callback
    # termino_p: termino multiplicativo que acompaña a kp
    # termino_i: termino integral que acompaña a ki
    # termino_d: termino derivatico que acompaña a kd
    # setpoint: Valor de tensión del setpoint
    # kp: constante multiplicativa del PID
    # ki: constante integral del PID
    # kd: constante derivativa del PID
    # isteps: cantidad de pasos para atrás utilizados para el termino integral        
    #####################################
    """
    
    # Valores maximos y minimos de duty cycle
    max_duty_cycle = 0.999
    min_duty_cycle = 0.001  
    
    # Cargo valores actuales
    setpoint = output_buffer_pid_constants[i,0]
    kp = output_buffer_pid_constants[i,1]
    ki = output_buffer_pid_constants[i,2]
    kd = output_buffer_pid_constants[i,3]
    isteps = int(output_buffer_pid_constants[i,4])
    output_buffer_error_data_i = output_buffer_error_data[i]
    mean_data_i = output_buffer_mean_data[i,0]
        
    # Paso anterior de buffer circular
    j = (i-1)%buffer_chunks
    
#        ## Ejemplo de cambio de parametros PID on the fly
#        setpoint = output_buffer_pid_constants[j,0] - 0.002
#        kd = output_buffer_pid_constants[j,3] + 0.002
#        output_buffer_error_data_i = mean_data_i - setpoint
            
    # isteps paso anterior de buffer circular
    k = (i-isteps)%buffer_chunks    
    
    # Algoritmo PID
    termino_p = output_buffer_error_data_i
    termino_d = output_buffer_error_data_i - output_buffer_error_data[j]
    termino_d = termino_d*sample_period
    
    # Termino integral (hay que optimizar esto)
    termino_i = 0
    if k >= i:
        termino_i = np.sum(output_buffer_error_data[k:buffer_chunks]) + np.sum(output_buffer_error_data[0:i])
    else:
        termino_i = np.sum(output_buffer_error_data[k:i])
    termino_i = termino_i*sample_period
    
    output_buffer_duty_cycle_i =  kp*termino_p + ki*termino_i + kd*termino_d

    # Salida de la función
    output_buffer_duty_cycle_i = min(output_buffer_duty_cycle_i,max_duty_cycle)
    output_buffer_duty_cycle_i = max(output_buffer_duty_cycle_i,min_duty_cycle)
    
    return output_buffer_duty_cycle_i, output_buffer_error_data_i, termino_p, termino_i, termino_d, setpoint, kp, ki, kd, isteps 






##
carpeta_salida = 'PID2'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
         
# Variables 
ai_channels = [4]
buffer_chunks = 2000
ai_samples = 500
ai_samplerate = 50000
do_channel = [0]
initial_do_duty_cycle = 0.5
initial_do_frequency = 2000
setpoint = 4.42
kp = 0.96
ki = 22.98
kd = 19.86
isteps = 60
path_data_save = os.path.join(carpeta_salida,'experimento_tt')
callback_pid_variables = {}



##
parametros = {}
parametros['buffer_chunks'] = buffer_chunks
parametros['ai_samples'] = ai_samples
parametros['ai_samplerate'] = ai_samplerate
parametros['ai_device'] = 'Dev2/ai'
parametros['ai_channels'] = ai_channels  
parametros['do_device'] = 'Dev2/ctr'
parametros['do_channel'] = do_channel  
parametros['initial_do_duty_cycle'] = initial_do_duty_cycle
parametros['initial_do_frequency'] = initial_do_frequency
parametros['setpoint'] = setpoint
parametros['pid_constants'] = [kp,ki,kd,isteps]
parametros['save_raw_data'] = False
parametros['save_processed_data'] = True
parametros['path_data_save'] = path_data_save
parametros['show_plot'] = True
parametros['callback_pid'] = callback_pid    
parametros['callback_pid_variables'] = callback_pid_variables

parametros['sub_chunk_save'] = 25
parametros['sub_chunk_plot'] = 25
parametros['nbr_buffers_plot'] = 2
parametros['plot_rate_hz'] = 13

adquisicion = pid_daq(parametros)
adquisicion.configure_acquisition()
        
        
#%%
    
def pid_daqmx(parametros):
    
    """
    Descripción del programa
    ------------------------
    
    Este programa utiliza la placa de adquisición de National Instruments USB 6210 para realizar un lazo de control PID.
    Para ello utiliza las entradas analógicas para la medición del sensor y la salida digital en modo PWM para el control del actuador.
    El control se realiza variando el duty cycle del tren de pulsos enviado. El programa utiliza un esquema multithreading para la realización 
    de las distintas tareas que se describen a continuación:
        - reader_thread: realiza la medición del sensor (chunk de ai_samples) y la escribe en una fila del buffer de entrada input_buffer (buffer de tiop circular).
        - callback_thread: realiza el procesamiento de la señal adquirida, obtiene el duty cycle de salida y la escribe en el output_buffer (buffer de tiop circular).
        - writer_thread: manda el duty cycle del PWM al instrumento, el cual está conectado al actuador
        - data_writer1_thread: escribe el dato crudo del buffer_input en un archivo de salida
        - data_writer2_thread: escribe el dato procesado por el thread callback y alojado en el output buffer en un archivo de salida
        - plot_thread: realiza el muestreo de los datos procesados en el thread callback.
        
    Para la comunicación entre los threads se utilizan los siguientes semaforos:
        - semaphore1: se incrementa cada vez que se recibe una medición (reader_thread) y decrementa cada vez que se lee un chunk en el callback_thread.
        - semaphore2: se incrementa cada vez que se termina de procesar un chunk en el callback_thread y se escribe el output_buffer, 
        y decrementa cada vez que se envia el duty cycle al instrumento en el writer_thread.
        - semaphore3: se incrementa cada vez que se recibe una medición (reader_thread) y decrementa cada vez que se escribe un chunk de dato crudo en el archivo de salida (data_writer1_thread).
        - semaphore4: se incrementa cada vez que se termina de procesar un chunk en el callback_thread y decrementa cada vez que se escribe el dato procesado en el archivo de salida (data_writer2_thread).
        - semaphore5: se incrementa cada vez que se termina de procesar un chunk en el callback_thread y decrementa cada vez que se reciben los datos procesados para el muestreo de los mismos (plot_thread).
        
    Cada vez que se verifica overrun de cualquier semáforo, en el vaciado y llenado de los buffers de entrada y salida, la medición se interrumpe y se avisa la interrupción.
    
    La entrada de los parámetros de adquisición se realiza con un diccionario con las siguientes variables:
        - buffer_chunks : int, cantidad de chunks de los buffers de entrada (input_buffer) y salida (output_buffer).
        - ai_samples : int, cantidad de samples por chunk.
        - ai_channels : lista de int de dos elementos, canales de medicion [canal_i, canal_f]. El control PID se realiza con el primer canal especificado. En caso de un canal, la lista es de un elemento [canal].
        - ai_samplerate : int, frecuencia de sampleo de los canales analógicos. Tener en cuenta las limitaciones del instrumento cuando se trabaja con varios canales.
        - ai_device : string, nombre del dispositivo analogico ej. 'Dev1/ai'
        - do_device : string, nombre del dispositivo digital ej. 'Dev1/ctr'
        - do_channel : lista de int de un elemento, [canal] por ahora solo esta habilitado un canal de salida.
        - initial_do_duty_cycle : int, duty cycle inicial del PWM
        - initial_do_frequency : int, frecuencia del PWM . La frecuencia no varía a lo largo de la medición.
        - setpoint : float, setpoint en [V].
        - pid_constants: lista de 4 elementos, 3 floats y 1 int. [kp,ki,kd,isteps]:
            kp : float, constante multiplicativa
            ki : float, constante integrativa
            kd : float, constante derivativa
            isteps : int, pasos para realizar el termino integral, debe ser menor a buffer_chunk.
        - save_raw_data : bool True o False, guarda el dato crudo
        - save_processed_data : bool True o False, guarda el dato procesado
        - path_data_save : string, path de directorio donde se guardan los resultados. El directorio debe ser creado antes.
        La primer dimensión corresponde al tiempo, es decir cada fila es un chunk. Se puede utilizar la función load_from_np_file(filename) para abrir los archivos.
        Los archivos de salida son los siguientes:
            path_data_save + '_raw_data.bin' : archivo binario para dato crudo. Es un array de tres dimensiones [:, ai_samples,ai_nbr_channels]
            path_data_save + '_duty_cycle.bin' : archivo binario con duty_cycle. Es un array de una dimensión.
            path_data_save + '_mean_data.bin' : archivo binario con valor medio de los canales. Es un array de dos dimensiones [:,ai_nbr_channels]
            path_data_save + '_pid_constants.bin' : archivo binario con las constantes PID. Es un un array de dos dimensiones [:,setpoint kp ki kp isteps]
            path_data_save + '_pid_terminos.bin' : archivo binario con los términos PID. Es un array de dos dimensiones [:, termino_p termino_i termino_p]                 
        - show_plot : bool, actualiza el muestreo o no.
        - callback_pid : function, función con el callback. Ver P2_corre_pid.py con ejemplo.
        - callback_pid_variables : lista, lista donde se colocan las variables (definidas por el usuario) que pudiera utilizar el callback.
        - sub_chunk_save : int, parámetro opcional. Especifica la cantidad de chunks que se guardan por vez.
        - sub_chunk_plot : int, parámetro opcional. Especifica la cantidad de chunks que se muestran por vez.
        - plot_rate_hz : int, parámetro opcional, pisa a sub_chunk_plot. Frecuencia de muestreo en Hz. Se aconseja frecuencias menores a 15 Hz. Hay que ver si con animation de matplotlib mejora.
        
        Al comenzar la adquisicón se abre la interfaz que permite visualizar la medición de los canales analógicos y el valor de duty cycle en el grafico1. Y los términos
        multiplicativos, integrales, y derivativos del PID en el grafico2. Tambíen se permite el cambio de las constantes kp, ki, kd y la cantidad de pasos utilizados en el término integral.
        
        Autores: Leslie Cusato, Marco Petriella
        Automatización y control - 2do cuatrimestre 2018
    """

    

    default_fontsize = 6
    plot_fontsize = 8
    
    # Errores 
    initialize_error = []
    acquiring_error = []  
    warnings = []
    evento_warning = threading.Event()
    evento_salida = threading.Event()
    
    ##### Callbacks de error #######
    def warning_callback(warning_string): 
        evento_warning.set()
        warnings.append(warning_string)  
        
    def exit_callback(event):  
        acquiring_error.append('Medición interrumpida por el usuario')
        evento_salida.set()

    def exit_callback1(error_string):   
        acquiring_error.append(error_string) 
        evento_salida.set()
                
    # Lectura de parametros
    buffer_chunks = parametros['buffer_chunks']   
    ai_samples = parametros['ai_samples']
    ai_samplerate = parametros['ai_samplerate']
    ai_device = parametros['ai_device']
    ai_channels = parametros['ai_channels']
    do_device = parametros['do_device']
    do_channel = parametros['do_channel']
    initial_do_duty_cycle = parametros['initial_do_duty_cycle']
    initial_do_frequency = parametros['initial_do_frequency']        
    setpoint = parametros['setpoint']
    [kp,ki,kd,isteps]  = parametros['pid_constants']
    save_raw_data = parametros['save_raw_data']
    save_processed_data = parametros['save_processed_data']
    show_plot = parametros['show_plot']
    path_data_save = parametros['path_data_save']      
    callback_pid = parametros['callback_pid']
    callback_pid_variables = parametros['callback_pid_variables']    
    
    # Path de los archivos de salida
    path_raw_data = path_data_save + '_raw_data.bin'
    path_duty_cycle_data = path_data_save + '_duty_cycle.bin'
    path_mean_data = path_data_save + '_mean_data.bin'
    path_pid_constants = path_data_save + '_pid_constants.bin'
    path_pid_terminos = path_data_save + '_pid_terminos.bin'
    
    if save_raw_data:
        if os.path.exists(path_raw_data):
            os.remove(path_raw_data)
            
    if save_processed_data:
        if os.path.exists(path_duty_cycle_data):
            os.remove(path_duty_cycle_data)
    
        if os.path.exists(path_mean_data):
            os.remove(path_mean_data)        

        if os.path.exists(path_pid_constants):
            os.remove(path_pid_constants)    

        if os.path.exists(path_pid_terminos):
            os.remove(path_pid_terminos)   
    
    
    ##### ACONDICIONAMIENTO DE PARAMETROS ###########
    #################################################
    # Sub chunks a guardar y graficar
    if 'sub_chunk_save' in parametros:
        sub_chunk_save = parametros['sub_chunk_save'] 
        if buffer_chunks%sub_chunk_save != 0:
            initialize_error.append('buffer_chunks debe ser multiplo de sub_chunk_save')                
    else:
        i = 20
        while i > 1:
            sub_chunk_save = buffer_chunks/i
            if buffer_chunks%i == 0:
                break
            i = i - 1
        if i == 1:
            initialize_error.append('No se encuentra sub_chunk_save tal que buffer_chunks sea multiplo')   
    sub_chunk_save = int(sub_chunk_save)  
        
    if 'sub_chunk_plot' in parametros:    
        sub_chunk_plot = parametros['sub_chunk_plot'] 
        if buffer_chunks%sub_chunk_plot != 0:
            initialize_error.append('buffer_chunks debe ser multiplo de sub_chunk_plot')    
    else:
        i = 20
        while i > 1:
            sub_chunk_plot = buffer_chunks/i
            if buffer_chunks%i == 0:
                break
            i = i - 1
        if i == 1:
            initialize_error.append('No se encuentra sub_chunk_plot tal que buffer_chunks sea multiplo')     
    
    if 'plot_rate_hz' in parametros:
        plot_rate_hz = parametros['plot_rate_hz']
        sub_chunk_plot = ai_samplerate/ai_samples/plot_rate_hz
        if buffer_chunks%sub_chunk_plot != 0 or sub_chunk_plot != int(sub_chunk_plot):
            warning_string = 'El plot_rate_hz indicado no es posible, se busca el más cercano'
            warning_callback(warning_string)
            sub_chunk_plot = int(sub_chunk_plot*1.5)
            while sub_chunk_plot > 1:
                if buffer_chunks%sub_chunk_plot == 0:
                    break
                sub_chunk_plot = int(sub_chunk_plot-1)
            if sub_chunk_plot == 1:
                initialize_error.append('No se encuentra sub_chunk_plot tal que buffer_chunks sea multiplo')  
    sub_chunk_plot = int(sub_chunk_plot)
    
    if (ai_samples/ai_samplerate)%(1/initial_do_frequency) != 0.:
        warning_string = 'La cantidad de ciclos del PWM no es entera!'
        warning_callback(warning_string)

    # pasos de la integral del pid
    paso_integral = 10
    possible_isteps = np.arange(0,buffer_chunks+paso_integral,paso_integral,dtype=int)
    possible_isteps[0] = 1

    # Largo del vector a graficar
    if 'nbr_buffers_plot' in parametros:
        nbr_buffers_plot = int(parametros['nbr_buffers_plot'])
    else:
        nbr_buffers_plot = 10
        
    # ai string
    ai_channels_str = ''
    ai_nbr_channels = 0
    if len(ai_channels) == 2:
        ai_channels_str = str(ai_channels[0]) + ':' + str(ai_channels[1])
        ai_nbr_channels = ai_channels[1] - ai_channels[0] + 1
    elif len(ai_channels) == 1:
        ai_channels_str = str(ai_channels[0])
        ai_nbr_channels = 1
    else:
        initialize_error.append('El formato de los canales no está bien especificado')  
        
    ai_channels_str = ai_device + ai_channels_str

    # do string
    do_channels_str = do_device + str(do_channel[0])    
    if len(do_channel) > 1:
        initialize_error.append('Por ahora solo está habilitado un solo canal digital')   

    ##### FIN DE ACONDICIONAMIENTO DE PARAMETROS ###########
    #################################################

    ##############################
    ####### INICIO PLOT ##########     
    data_plot1 = np.zeros([buffer_chunks*nbr_buffers_plot,ai_nbr_channels])
    data_plot2 = np.zeros(buffer_chunks*nbr_buffers_plot)    
    data_plot3 = np.zeros([buffer_chunks*nbr_buffers_plot,3])  
    
    tiempo = np.arange(0,data_plot1.shape[0])/data_plot1.shape[0]
    tiempo = tiempo*(ai_samples*buffer_chunks*nbr_buffers_plot/ai_samplerate)
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d %H:%M:%S")
  
    fig = plt.figure(figsize=(7,3.7),dpi=250)
    ax = fig.add_axes([.08, .35, .70, .33])  
    ax1 = ax.twinx()
    
    line1 = []
    for i in range(ai_nbr_channels):
        line, = ax.plot(tiempo,data_plot1[:,i], '-')  
        line1.append(line)
    line2, = ax1.plot(tiempo,data_plot2, '-',color='red')
    text_now = ax.text(1.01,1.1,now,fontsize=default_fontsize,transform = ax.transAxes)
    ax.set_ylim([4,5])
    ax1.set_ylim([0,1.2])
    ax.set_xlabel('tiempo [s]',fontsize = plot_fontsize)
    ax1.set_ylabel('duty cycle',fontsize = plot_fontsize)
    ax.set_ylabel('mean [V]',fontsize = plot_fontsize)   
    setpoint_line = ax.axhline(setpoint,linestyle='--',linewidth=0.8)
    
    ax.xaxis.set_tick_params(labelsize=plot_fontsize)
    ax1.xaxis.set_tick_params(labelsize=plot_fontsize)
    ax.yaxis.set_tick_params(labelsize=plot_fontsize)
    ax1.yaxis.set_tick_params(labelsize=plot_fontsize)   
    ax.grid(linestyle='--',linewidth=0.3)   

    ##
    ax3 = fig.add_axes([.08, .71, .70, .27])  
    ax3.set_xticklabels([])
    line3 = []
    for i in range(data_plot3.shape[1]):
        line, = ax3.plot(tiempo,data_plot3[:,i], '-')  
        line3.append(line) 
    ax3.set_ylim([-2,2])
    ax3.legend(['P','I','D'],bbox_to_anchor=(1.01, 1.0))
    
    ax3.xaxis.set_tick_params(labelsize=plot_fontsize)
    ax3.yaxis.set_tick_params(labelsize=plot_fontsize) 
    ax3.grid(linestyle='--',linewidth=0.3)    
    
    ##         
    ax2 = fig.add_axes([.15, .03, .3, .3])        
    ax2.axis('off')
    xi = 0.68
    yi = 0.65
    dyi = 0.10
    ax2.text(xi,yi,'Pending processes',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
    ax2.text(xi,yi - 1*dyi,'Input buffer filling: ',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
    ax2.text(xi,yi - 2*dyi,'Output buffer emptying:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
    ax2.text(xi,yi - 3*dyi,'Raw data writer:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
    ax2.text(xi,yi - 4*dyi,'Processed data writer:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
    ax2.text(xi,yi - 5*dyi,'Plotting:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
    ax2.text(xi,yi - 6*dyi,'Measuring acquiring ratio:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
    
    xi = 1.53
    txt1 = ax2.text(xi,yi - 1*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
    txt2 = ax2.text(xi,yi - 2*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
    txt3 = ax2.text(xi,yi - 3*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
    txt4 = ax2.text(xi,yi - 4*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
    txt5 = ax2.text(xi,yi - 5*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
    txt6 = ax2.text(xi,yi - 6*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
    
    xi = -0.40
    ax2.text(xi,yi,'Acquisition parameters',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
    ax2.text(xi,yi - 1*dyi,'Samplerate: ',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
    ax2.text(xi,yi - 2*dyi,'Samples per chunk:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
    ax2.text(xi,yi - 3*dyi,'Nbr. chunks:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
    ax2.text(xi,yi - 4*dyi,'PWM frequency:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
    ax2.text(xi,yi - 5*dyi,'Nbr. PWM cycles per chunk:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
    ax2.text(xi,yi - 6*dyi,'Save raw / processed data:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
    ax2.text(xi,yi - 7*dyi,'Display plot chunks / rate:',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
    
    xi = 0.55
    ax2.text(xi,yi - 1*dyi,'%6.2f' % (ai_samplerate/1000.0) + ' kHz',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
    ax2.text(xi,yi - 2*dyi,'%4d' % ai_samples + ' / ' + '%6.2f' % (ai_samples/ai_samplerate*1000.) + ' ms',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
    ax2.text(xi,yi - 3*dyi,'%4d' % buffer_chunks + ' / ' + '%6.2f' % (buffer_chunks*ai_samples/ai_samplerate) + ' s',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
    ax2.text(xi,yi - 4*dyi,'%6.2f' % (initial_do_frequency/1000.) + ' kHz',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
    ax2.text(xi,yi - 5*dyi,'%6.2f' % ((ai_samples/ai_samplerate)*(initial_do_frequency)) ,fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
    ax2.text(xi,yi - 6*dyi, str(save_raw_data) + ' / ' + str(save_processed_data),fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)
    ax2.text(xi,yi - 7*dyi, '%4d' % sub_chunk_plot + ' / ' +'%6.2f' % (ai_samplerate/ai_samples/sub_chunk_plot) + ' Hz',fontsize=default_fontsize,va='center',ha='right',transform = ax2.transAxes)

    xi = 2.35
    ax2.text(xi,yi,'PID parameters',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
    pid_para0_txt = ax2.text(xi,yi - 1*dyi,'%6.2f' % (setpoint) + ' V',fontsize=default_fontsize,va='center',ha='left',transform = ax2.transAxes)
    pid_para1_txt = ax2.text(xi,yi - 2*dyi,'%6.2f' % (kp),fontsize=default_fontsize,va='center',ha='left',transform = ax2.transAxes)
    pid_para2_txt = ax2.text(xi,yi - 3*dyi,'%6.2f' % (ki),fontsize=default_fontsize,va='center',ha='left',transform = ax2.transAxes)
    pid_para3_txt = ax2.text(xi,yi - 4*dyi,'%6.2f' % (kd),fontsize=default_fontsize,va='center',ha='left',transform = ax2.transAxes)
    pid_para4_txt = ax2.text(xi+0.03,yi - 5*dyi,'%2d' % (isteps),fontsize=default_fontsize,va='center',ha='left',transform = ax2.transAxes)

    xi = 1.63
    ax2.text(xi,yi,'PID parameters',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
    
    xi_s = 0.69
    yi_s = 0.185
    dyi_s = 0.03
    
    
    # Mensje de error
    x_error = 1.64
    y_error = 0.01

    texto_error = ax2.text(x_error,y_error,'',fontsize=default_fontsize-1,va='center',transform = ax2.transAxes,color='red') 
    
    def print_error(string_list):       
        s_tot = ''
        for i in range(len(string_list)):
            s_tot = s_tot + string_list[i] + '\n' 
        texto_error.set_text(s_tot)       

    ############### BOTONES Y SLIDERS #########################
    
    global ssetpoint, skp, ski, skd,sisteps, lsetpoint, lkp, lki, lkd, listeps, bonoff, bnext, pid_onoff_button
    ########### CALLBACK DE BOTONES ##############
    def update_pid(val):
        global lsetpoint, lkp, lki, lkd, listeps
        lsetpoint = [ssetpoint.val]
        lkp = [skp.val]
        lki = [ski.val]
        lkd = [skd.val]  
        listeps = [sisteps.val]      
        ind_is = np.argmin(np.abs(possible_isteps - sisteps.val))       
        listeps = [possible_isteps[ind_is]]
        while sisteps.val != possible_isteps[ind_is]:
            sisteps.set_val(listeps[0])
        
    def pid_onoff(event):
        global pid_onoff_button
        if pid_onoff_button[0] is True:
            pid_onoff_button[0] = False
            bonoff.label.set_text('PID OFF')
            bonoff.label.set_color([0.99,0.0,0.])
        else:
            pid_onoff_button[0] = True  
            bonoff.label.set_text('PID ON ')
            bonoff.label.set_color([0.,0.99,0.])
            
    ########### FIN CALLBACK DE BOTONES ############## 
    # Inicializo variables de interfaz: setpoint, kp, ki, kd, isteps, pid_on_off
    lsetpoint = [setpoint]
    lkp = [kp]
    lki = [ki]
    lkd = [kd]
    listeps = [isteps]
    pid_onoff_button = [True]
    ##############################################################################    

    ############### BOTONES ######################
    # Boton de salida
    axnext = plt.axes([0.88, 0.35, 0.1, 0.075])
    bnext = Button(axnext, 'Stop')
    bnext.on_clicked(exit_callback) 
    
    axonoff = plt.axes([0.88, 0.25, 0.1, 0.075])
    bonoff = Button(axonoff, 'PID ON ')
    bonoff.on_clicked(pid_onoff)  
    bonoff.label.set_size(10)
    bonoff.label.set_backgroundcolor([1,1,1,0.8])
    bonoff.label.set_color([0.,0.99,0.])
    
    # Slider setpoit
    axcolor = 'lightgoldenrodyellow'
    axsetpoint = plt.axes([xi_s, yi_s, 0.12, 0.02], facecolor=axcolor)
    ssetpoint = Slider(axsetpoint, 'Setpoint',0.001, 5.0, valinit=setpoint)
    ssetpoint.on_changed(update_pid) 
    ssetpoint.label.set_size(default_fontsize)
    ssetpoint.valtext.set_size(default_fontsize)

    # Slider kp
    axkp = plt.axes([xi_s, yi_s - 1*dyi_s, 0.12, 0.02], facecolor=axcolor)
    skp = Slider(axkp, 'kp',0.0, 5.0, valinit=kp)
    skp.on_changed(update_pid) 
    skp.label.set_size(default_fontsize)
    skp.valtext.set_size(default_fontsize)

    # Slider ki
    axki = plt.axes([xi_s, yi_s - 2*dyi_s, 0.12, 0.02], facecolor=axcolor)
    ski = Slider(axki, 'ki',0.0, 50.0, valinit=ki)
    ski.on_changed(update_pid) 
    ski.label.set_size(default_fontsize)
    ski.valtext.set_size(default_fontsize)
    
    # Slider kd
    axkd = plt.axes([xi_s, yi_s - 3*dyi_s, 0.12, 0.02], facecolor=axcolor)
    skd = Slider(axkd, 'kd',0.0, 100.0, valinit=kd)
    skd.on_changed(update_pid)   
    skd.label.set_size(default_fontsize)
    skd.valtext.set_size(default_fontsize)        

    # Slider integral steps
    axisteps = plt.axes([xi_s, yi_s - 4*dyi_s, 0.12, 0.02], facecolor=axcolor)
    sisteps = Slider(axisteps, 'steps',0, buffer_chunks, valinit=isteps)
    sisteps.valfmt = '%2d'
    sisteps.on_changed(update_pid) 
    sisteps.label.set_size(default_fontsize)
    sisteps.valtext.set_size(default_fontsize)
      
    ############### FIN BOTONES ######################     
     
    
    ####### FIN PLOT ############
    #############################    


    # Defino los buffers de entrada y salida
    input_buffer = np.zeros([buffer_chunks,ai_samples,ai_nbr_channels])
    output_buffer_mean_data = np.zeros([buffer_chunks,ai_nbr_channels])
    output_buffer_duty_cycle = np.ones(buffer_chunks)*initial_do_duty_cycle
    output_buffer_error_data = np.zeros(buffer_chunks)
    output_buffer_pid_constants = np.ones([buffer_chunks,5]) 
    output_buffer_pid_constants[:,0] = output_buffer_pid_constants[:,0]*setpoint
    output_buffer_pid_constants[:,1] = output_buffer_pid_constants[:,1]*kp
    output_buffer_pid_constants[:,2] = output_buffer_pid_constants[:,2]*ki
    output_buffer_pid_constants[:,3] = output_buffer_pid_constants[:,3]*kd
    output_buffer_pid_constants[:,4] = output_buffer_pid_constants[:,4]*isteps
    output_buffer_pid_terminos = np.zeros([buffer_chunks,3])    
       
    # Semaforos
    semaphore1 = threading.Semaphore(0) # Input buffer
    semaphore2 = threading.Semaphore(0) # Output buffer
    semaphore3 = threading.Semaphore(0) # Guardado de raw data
    semaphore4 = threading.Semaphore(0) # Guardado de processed data
    semaphore5 = threading.Semaphore(0) # Plot
    
    sample_period = ai_samples/ai_samplerate

    ############### DEFINICION DE LOS THREADS #################
    ###########################################################
    
    # Defino el thread que envia la señal          
    def writer_thread():  
        
        with nidaqmx.Task() as task_do:
            
            task_do.co_channels.add_co_pulse_chan_freq(counter=do_channels_str,duty_cycle=initial_do_duty_cycle,freq=initial_do_frequency,units=nidaqmx.constants.FrequencyUnits.HZ)
            task_do.timing.cfg_implicit_timing(sample_mode=constants.AcquisitionType.CONTINUOUS)    
            digi_s = nidaqmx.stream_writers.CounterWriter(task_do.out_stream)
            task_do.start()
                       
            i = 0
            while not evento_salida.is_set():
                
                semaphore2.acquire()   
    
                digi_s.write_one_sample_pulse_frequency(frequency = initial_do_frequency, duty_cycle = output_buffer_duty_cycle[i])
                
                i = i+1
                i = i%buffer_chunks     

        
                       
#        i = 0
#        while not evento_salida.is_set():
#            
#            semaphore2.acquire()   
#            #time.sleep(0.02)
#            #digi_s.write_one_sample_pulse_frequency(frequency = initial_do_frequency, duty_cycle = output_buffer_duty_cycle[i])
#            
#            i = i+1
#            i = i%buffer_chunks           
               
    # Defino el thread que adquiere la señal   
    def reader_thread():
                
        with nidaqmx.Task() as task_ai:
            task_ai.ai_channels.add_ai_voltage_chan(ai_channels_str,max_val=5., min_val=-5.,terminal_config=constants.TerminalConfiguration.RSE)#, "Voltage")#,AIVoltageUnits.Volts)#,max_val=10., min_val=-10.)
            task_ai.timing.cfg_samp_clk_timing(ai_samplerate,samps_per_chan=ai_samples,sample_mode=constants.AcquisitionType.CONTINUOUS)
                
            i = 0
            while not evento_salida.is_set():
        
                medicion = task_ai.read(number_of_samples_per_channel=ai_samples)
                medicion = np.asarray(medicion)
                medicion = np.reshape(medicion,ai_nbr_channels*ai_samples,order='F')
                
                for j in range(ai_nbr_channels):
                    input_buffer[i,:,j] = medicion[j::ai_nbr_channels]  
                
                semaphore1.release() 
                semaphore3.release()
                
                i = i+1
                i = i%buffer_chunks                  

#        i = 0
#        while not evento_salida.is_set():
#    
#            medicion = np.zeros([ai_nbr_channels,ai_samples])
#            #medicion[0,:] = np.arange(0,ai_samples)
#            tt = i*ai_samples/ai_samplerate + np.arange(ai_samples)/ai_samples/ai_samplerate
#            medicion[0,:] = 2.1 + 1.0*np.sin(2*np.pi*0.2*tt) + np.random.rand(ai_samples)
#            #medicion[1,:] = 1 + np.random.rand(ai_samples)
#            medicion = np.reshape(medicion,ai_nbr_channels*ai_samples,order='F')
#            
#            for j in range(ai_nbr_channels):
#                input_buffer[i,:,j] = medicion[j::ai_nbr_channels]  
#            
#            semaphore1.release() 
#            semaphore3.release()
#            
#            time.sleep(0.009)
#            
#            i = i+1
#            i = i%buffer_chunks  

       
    # Thread del callback        
    def callback_thread():
                
        i = 0
        while not evento_salida.is_set(): 
            

            if semaphore1._value > buffer_chunks:
                error_string = 'Hay overrun en llenado del input_buffer!'
                exit_callback1(error_string)        
                            
            if semaphore2._value > buffer_chunks:
                error_string = 'Hay overrun en el vaciado del output_buffer!'
                exit_callback1(error_string)            
            
            semaphore1.acquire() 
            
#            # Paso anterior del buffer circular
#            j = (i-1)%buffer_chunks
    
            ## Inicio Callback      
            output_buffer_mean_data[i,:] = np.mean(input_buffer[i,:,:],axis=0)
            output_buffer_error_data[i] = output_buffer_mean_data[i,0] - lsetpoint[0]              
            output_buffer_pid_constants[i,:] = np.array([lsetpoint[0],lkp[0],lki[0],lkd[0],listeps[0]])
         
            # funcion de callback
            output_buffer_duty_cycle_i, output_buffer_error_data_i, termino_p, termino_i, termino_d, setpoint, kp, ki, kd, isteps  = callback_pid(i, input_buffer, output_buffer_duty_cycle, output_buffer_pid_terminos, output_buffer_mean_data, output_buffer_error_data, output_buffer_pid_constants, buffer_chunks, sample_period ,callback_pid_variables)             
            
            # Actualizo los buffers luego del callback
            if pid_onoff_button[0] is False:
                output_buffer_duty_cycle_i = initial_do_duty_cycle

            output_buffer_duty_cycle[i] = output_buffer_duty_cycle_i       
            output_buffer_pid_terminos[i,:] = np.array([termino_p, termino_i, termino_d])            
            output_buffer_pid_constants[i,:] = np.array([setpoint,kp,ki,kd,isteps])    
            output_buffer_error_data[i] = output_buffer_error_data_i            
            ## Fin callback
            
            semaphore2.release()
            semaphore4.release()
            semaphore5.release()                                
           
            i = i+1
            i = i%buffer_chunks   
       

    def data_writer1_thread(save_raw_data):
                
        i = 0
        
        if not save_raw_data:
            while not evento_salida.is_set():  
                semaphore3.acquire() 
                
        else:
                        
            while not evento_salida.is_set():  
    
                if semaphore3._value > buffer_chunks:
                    error_string = 'Hay overrun en la escritura de datos raw data!'
                    exit_callback1(error_string)               
                
                semaphore3.acquire() 
                
                if not i%sub_chunk_save: 
                    
                    j = (i-sub_chunk_save)%buffer_chunks  
                    jj = (j+sub_chunk_save-1)%buffer_chunks + 1
                    
                    try:
                        save_to_np_file(path_raw_data,input_buffer[j:jj,:,:]) 
                    except:
                        warning_string = 'Error: No se guarda raw data'
                        warning_callback(warning_string)
               
                i = i+1
                i = i%buffer_chunks   

            
    def data_writer2_thread(save_processed_data):
                
        i = 0
        
        if not save_processed_data:
            while evento_salida.is_set():  
                semaphore4.acquire() 
        else:
    
            while not evento_salida.is_set(): 
    
                if semaphore4._value > buffer_chunks:    
                    error_string = 'Hay overrun en la escritura de datos duty cycle!'
                    exit_callback1(error_string)                          
                
                semaphore4.acquire() 
                
                if not i%sub_chunk_save:
                     
                    j = (i-sub_chunk_save)%buffer_chunks  
                    jj = (j+sub_chunk_save-1)%buffer_chunks + 1 
                    
                    try:
                        save_to_np_file(path_duty_cycle_data, output_buffer_duty_cycle[j:jj]) 
                    except:
                        warning_string = 'Error: No se guarda duty cycle'
                        warning_callback(warning_string)                        
    
                    try:
                        save_to_np_file(path_mean_data, output_buffer_mean_data[j:jj]) 
                    except:
                        warning_string = 'Error: No se guarda mean data'
                        warning_callback(warning_string)                             
    
                    try:
                        save_to_np_file(path_pid_constants, output_buffer_pid_constants[j:jj,:]) 
                    except:
                        warning_string = 'Error: No se guardan las constantes pid'
                        warning_callback(warning_string)                            
    
                    try:
                        save_to_np_file(path_pid_terminos, output_buffer_pid_terminos[j:jj,:]) 
                    except:
                        warning_string = 'Error: No se guardan los terminos pid'
                        warning_callback(warning_string)                           
                                            
                i = i+1
                i = i%buffer_chunks             
        
    
    def plot_thread():
        global warnings
        global ssetpoint, skp, ski, skd,sisteps, lsetpoint, lkp, lki, lkd, listeps, bonoff, bnext
                    
        
        lsetpoint = [setpoint]
         
        # Contador de los semaforos
        x_semaphores = np.zeros(5)
        
        # Para el plot
        data_plot1 = np.zeros([buffer_chunks*nbr_buffers_plot,ai_nbr_channels])
        data_plot2 = np.zeros(buffer_chunks*nbr_buffers_plot)                
        data_plot3 = np.zeros([buffer_chunks*nbr_buffers_plot,3])
        
        # Para tiempo de medicion y de adquisicion
        previous = datetime.datetime.now()
        delta_t = np.array([])
        delta_t_avg = 10
        measure_adq_ratio = 0   
           
                
        i = 0
        while not evento_salida.is_set(): 
        
            if semaphore5._value > buffer_chunks:
                    error_string = 'Hay overrun en el plot!'
                    exit_callback1(error_string)         
            
            semaphore5.acquire()
            
            if not show_plot:
                continue
            
            if i%sub_chunk_plot == 0: 
                
                j = (i-sub_chunk_plot)%buffer_chunks  
                jj = (j+sub_chunk_plot-1)%buffer_chunks + 1 
    
                # Medición de tiempos
                now = datetime.datetime.now()
                delta_ti = now - previous
                previous = now
                now = now.strftime("%Y-%m-%d %H:%M:%S")
                
                delta_ti = delta_ti.total_seconds()
                delta_t = np.append(delta_t,delta_ti)
                if delta_t.shape[0] > delta_t_avg:
                    delta_t = delta_t[-delta_t_avg:]
               
                if delta_t.mean() > 0:
                    measure_adq_ratio = (ai_samples/ai_samplerate*sub_chunk_plot)/delta_t.mean()
           
                # Data update
                data_plot1[0:-sub_chunk_plot,:] = data_plot1[sub_chunk_plot:,:]
                data_plot1[-sub_chunk_plot:,:] = output_buffer_mean_data[j:jj,:]
                
                data_plot2[0:-sub_chunk_plot] = data_plot2[sub_chunk_plot:]   
                data_plot2[-sub_chunk_plot:] = output_buffer_duty_cycle[j:jj]
                                         
                for k in range(ai_nbr_channels): 
                    line1[k].set_ydata(data_plot1[:,k])                    
                line2.set_ydata(data_plot2) 
    
                setpoint_line.set_ydata(output_buffer_pid_constants[i,0])
                
                data_plot3[0:-sub_chunk_plot,:] = data_plot3[sub_chunk_plot:,:]
                data_plot3[-sub_chunk_plot:,:] = output_buffer_pid_terminos[j:jj,:]
                for k in range(data_plot3.shape[1]): 
                    line3[k].set_ydata(data_plot3[:,k])             
 
                # Textos
                text_now.set_text(now)
                
                # BUffer
                x_semaphores[0] = (semaphore1._value/buffer_chunks)*100
                x_semaphores[1] = (semaphore2._value/buffer_chunks)*100
                x_semaphores[2] = (semaphore3._value/buffer_chunks)*100
                x_semaphores[3] = (semaphore4._value/buffer_chunks)*100
                x_semaphores[4] = (semaphore5._value/buffer_chunks)*100
                
                txt1.set_text('%2d' % x_semaphores[0] + ' %')
                txt2.set_text('%2d' % x_semaphores[1] + ' %')
                txt3.set_text('%2d' % x_semaphores[2] + ' %')
                txt4.set_text('%2d' % x_semaphores[3] + ' %')
                txt5.set_text('%2d' % x_semaphores[4] + ' %')              
                txt6.set_text('%4.2f' % measure_adq_ratio)
#                
                pid_para0_txt.set_text('%6.2f' % output_buffer_pid_constants[i,0] + ' V')
                pid_para1_txt.set_text('%6.2f' %output_buffer_pid_constants[i,1])
                pid_para2_txt.set_text('%6.2f' %output_buffer_pid_constants[i,2])
                pid_para3_txt.set_text('%6.2f' %output_buffer_pid_constants[i,3])
                pid_para4_txt.set_text('%2d' %output_buffer_pid_constants[i,4])
                                     
                
                if evento_warning.is_set():                
                    print_error(warnings)
                    warnings = []
                    evento_warning.clear() 
                    
                fig.canvas.draw_idle()
                   
            i = i+1
            i = i%buffer_chunks 
            
        print_error(acquiring_error)
        fig.canvas.draw_idle()
        
#        time.sleep(0.5)
#        print_error(['Cerrando interfaz...'])
#        time.sleep(0.5)
#        plt.close(fig)
 
    # Inicio los threads    
    t1 = threading.Thread(target=writer_thread, args=[])
    t2 = threading.Thread(target=reader_thread, args=[])
    t3 = threading.Thread(target=callback_thread, args=[])
    t4 = threading.Thread(target=data_writer1_thread, args=[save_raw_data])
    t5 = threading.Thread(target=data_writer2_thread, args=[save_processed_data])
    t6 = threading.Thread(target=plot_thread, args=[])

    # Imprimo warnings           
    if evento_warning.is_set():
        print_error(warnings)
        evento_warning.clear()
        warnings = []
    
    if len(initialize_error) == 0:
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t5.start()
        t6.start()
    else:
        initialize_error = [u'Error de inicialización'] + initialize_error
        print_error(initialize_error)
        
