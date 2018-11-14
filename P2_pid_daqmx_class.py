
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

import nidaqmx
import nidaqmx.constants as constants
import nidaqmx.stream_writers

class pid_daq(object):

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
    
    def __init__(self, parametros):       
        self.parametros = parametros
        
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
        self.ai_voltage_range = parametros['ai_voltage_range']
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
        self.sample_period = self.ai_samples/self.ai_samplerate
        
        # Valores predeterminado
        self.pid_onoff_button = True
        self.pid_costants_button = False
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
        
        self.ax1_lim_i = 0
        self.ax1_lim_f = 5
        self.ax2_lim_i = 0
        self.ax2_lim_f = 1.2
        self.ax3_lim_i = -2
        self.ax3_lim_f = 2 
        
        # Overrun de semaforos
        buffer_chunks = self.buffer_chunks
        self.semaforo1_ovr = buffer_chunks
        self.semaforo2_ovr = buffer_chunks
        self.semaforo3_ovr = buffer_chunks
        self.semaforo4_ovr = buffer_chunks
        self.semaforo5_ovr = buffer_chunks
        
        # Valores de la placa de adquisicion
        self.max_duty_cycle = 0.999
        self.min_duty_cycle = 0.001  
                     
        
    def acondiciona_variables(self):
        # Archivos de salida
        self.path_raw_data = self.path_data_save + '_raw_data.bin'
        self.path_duty_cycle_data = self.path_data_save + '_duty_cycle.bin'
        self.path_mean_data = self.path_data_save + '_mean_data.bin'
        self.path_pid_constants = self.path_data_save + '_pid_constants.bin'
        self.path_pid_terminos = self.path_data_save + '_pid_terminos.bin'
        self.path_semaforos = self.path_data_save + '_semaforos.bin'
        self.path_timestamp = self.path_data_save + '_timestamp.txt'
        self.path_parametros = self.path_data_save + '_parametros_iniciales.txt'
        
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

            if os.path.exists(self.path_semaforos):
                os.remove(self.path_semaforos)  

            if os.path.exists(self.path_timestamp):
                os.remove(self.path_timestamp)  
                
            if os.path.exists(self.path_parametros):
                os.remove(self.path_parametros)  
        
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
        output_buffer_duty_cycle = np.ones([buffer_chunks,1])*initial_do_duty_cycle
        output_buffer_error_data = np.zeros([buffer_chunks,1])
        output_buffer_pid_constants = np.ones([buffer_chunks,5]) 
        output_buffer_pid_constants[:,0] = output_buffer_pid_constants[:,0]*setpoint
        output_buffer_pid_constants[:,1] = output_buffer_pid_constants[:,1]*kp
        output_buffer_pid_constants[:,2] = output_buffer_pid_constants[:,2]*ki
        output_buffer_pid_constants[:,3] = output_buffer_pid_constants[:,3]*kd
        output_buffer_pid_constants[:,4] = output_buffer_pid_constants[:,4]*isteps
        output_buffer_pid_terminos = np.zeros([buffer_chunks,3])      
        output_buffer_semaforos = np.zeros([buffer_chunks,5])   
        output_buffer_timestamp = []
        for i in range(buffer_chunks):
            output_buffer_timestamp.append(0)
        
        self.input_buffer = input_buffer
        self.output_buffer_mean_data = output_buffer_mean_data
        self.output_buffer_duty_cycle = output_buffer_duty_cycle
        self.output_buffer_error_data = output_buffer_error_data
        self.output_buffer_pid_constants = output_buffer_pid_constants
        self.output_buffer_pid_terminos = output_buffer_pid_terminos
        self.output_buffer_semaforos = output_buffer_semaforos
        self.output_buffer_timestamp = output_buffer_timestamp

 
    def inicializa_interfaz(self):

        default_fontsize = self.default_fontsize
        plot_fontsize = self.plot_fontsize
        ax1_lim_i = self.ax1_lim_i
        ax1_lim_f = self.ax1_lim_f
        ax2_lim_i = self.ax2_lim_i
        ax2_lim_f = self.ax2_lim_f
        ax3_lim_i = self.ax3_lim_i
        ax3_lim_f = self.ax3_lim_f        
        
        
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
        data_plot2 = np.zeros([buffer_chunks*nbr_buffers_plot,1])    
        data_plot3 = np.zeros([buffer_chunks*nbr_buffers_plot,3])  
        
        tiempo_vec = np.arange(0,data_plot1.shape[0])/data_plot1.shape[0]
        tiempo_vec = tiempo_vec*(ai_samples*buffer_chunks*nbr_buffers_plot/ai_samplerate)
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")
      
        fig = plt.figure(figsize=(7,3.7),dpi=250)
        ax1 = fig.add_axes([.10, .35, .70, .33])  
        ax2 = ax1.twinx()
        
        line1 = []
        for i in range(data_plot1.shape[1]):
            line, = ax1.plot(tiempo_vec,data_plot1[:,i], '-')  
            line1.append(line)
        ax1.set_ylim([ax1_lim_i,ax1_lim_f])
        ax1.set_xlabel('tiempo [s]',fontsize = plot_fontsize)
        ax1.set_ylabel('mean [V]',fontsize = plot_fontsize)  
        ax1.xaxis.set_tick_params(labelsize=plot_fontsize)    
        ax1.yaxis.set_tick_params(labelsize=plot_fontsize)         
        ax1.grid(linestyle='--',linewidth=0.3)       
        
        line2 = []
        for i in range(data_plot2.shape[1]):
            line, = ax2.plot(tiempo_vec,data_plot2[:,i], '-',color='red')  
            line2.append(line)
        ax2.set_ylim([ax2_lim_i,ax2_lim_f])
        ax2.set_ylabel('duty cycle',fontsize = plot_fontsize)
        ax2.xaxis.set_tick_params(labelsize=plot_fontsize)
        ax2.yaxis.set_tick_params(labelsize=plot_fontsize)  
        
        setpoint_line = ax1.axhline(setpoint,linestyle='--',linewidth=0.8)
        txt_now = ax1.text(1.01,1.1,now,fontsize=default_fontsize,transform = ax1.transAxes)
    
        ##
        ax3 = fig.add_axes([.10, .71, .70, .27])  
        ax3.set_xticklabels([])
        line3 = []
        for i in range(data_plot3.shape[1]):
            line, = ax3.plot(tiempo_vec,data_plot3[:,i], '-')  
            line3.append(line) 
        ax3.set_ylim([ax3_lim_i,ax3_lim_f])
        ax3.legend(['P','I','D'],bbox_to_anchor=(1.13, 1.0),fontsize=9)
        
        ax3.xaxis.set_tick_params(labelsize=plot_fontsize)
        ax3.yaxis.set_tick_params(labelsize=plot_fontsize) 
        ax3.grid(linestyle='--',linewidth=0.3)    
        
        ##         
        ax4 = fig.add_axes([.15, .03, .3, .3])        
        ax4.axis('off')
        xi = 0.68
        yi = 0.65
        dyi = 0.10
        ax4.text(xi,yi,'Pending processes',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
        ax4.text(xi,yi - 1*dyi,'Input buffer filling: ',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
        ax4.text(xi,yi - 2*dyi,'Output buffer emptying:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
        ax4.text(xi,yi - 3*dyi,'Raw data writer:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
        ax4.text(xi,yi - 4*dyi,'Processed data writer:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
        ax4.text(xi,yi - 5*dyi,'Plotting:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
        ax4.text(xi,yi - 6*dyi,'Measuring acquiring ratio:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
        
        xi = 1.53
        monitoring_txt0 = ax4.text(xi,yi - 1*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
        monitoring_txt1 = ax4.text(xi,yi - 2*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
        monitoring_txt2 = ax4.text(xi,yi - 3*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
        monitoring_txt3 = ax4.text(xi,yi - 4*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
        monitoring_txt4 = ax4.text(xi,yi - 5*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
        monitoring_txt5 = ax4.text(xi,yi - 6*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
        
        xi = -0.40
        ax4.text(xi,yi,'Acquisition parameters',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
        ax4.text(xi,yi - 1*dyi,'Samplerate: ',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
        ax4.text(xi,yi - 2*dyi,'Samples per chunk:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
        ax4.text(xi,yi - 3*dyi,'Nbr. chunks:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
        ax4.text(xi,yi - 4*dyi,'PWM frequency:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
        ax4.text(xi,yi - 5*dyi,'Nbr. PWM cycles per chunk:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
        ax4.text(xi,yi - 6*dyi,'Save raw / processed data:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
        ax4.text(xi,yi - 7*dyi,'Display plot chunks / rate:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
        
        xi = 0.55
        ax4.text(xi,yi - 1*dyi,'%6.2f' % (ai_samplerate/1000.0) + ' kHz',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
        ax4.text(xi,yi - 2*dyi,'%4d' % ai_samples + ' / ' + '%6.2f' % (ai_samples/ai_samplerate*1000.) + ' ms',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
        ax4.text(xi,yi - 3*dyi,'%4d' % buffer_chunks + ' / ' + '%6.2f' % (buffer_chunks*ai_samples/ai_samplerate) + ' s',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
        ax4.text(xi,yi - 4*dyi,'%6.2f' % (initial_do_frequency/1000.) + ' kHz',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
        ax4.text(xi,yi - 5*dyi,'%6.2f' % ((ai_samples/ai_samplerate)*(initial_do_frequency)) ,fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
        ax4.text(xi,yi - 6*dyi, str(save_raw_data) + ' / ' + str(save_processed_data),fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
        ax4.text(xi,yi - 7*dyi, '%4d' % sub_chunk_plot + ' / ' +'%6.2f' % (ai_samplerate/ai_samples/sub_chunk_plot) + ' Hz',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
    
        xi = 2.35
        txt_pid_parameters = []
        ax4.text(xi,yi,'PID parameters',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
        pid_para_txt0 = ax4.text(xi,yi - 1*dyi,'%6.2f' % (setpoint) + ' V',fontsize=default_fontsize,va='center',ha='left',transform = ax4.transAxes)
        pid_para_txt1 = ax4.text(xi,yi - 2*dyi,'%6.2f' % (kp),fontsize=default_fontsize,va='center',ha='left',transform = ax4.transAxes)
        pid_para_txt2 = ax4.text(xi,yi - 3*dyi,'%6.2f' % (ki),fontsize=default_fontsize,va='center',ha='left',transform = ax4.transAxes)
        pid_para_txt3 = ax4.text(xi,yi - 4*dyi,'%6.2f' % (kd),fontsize=default_fontsize,va='center',ha='left',transform = ax4.transAxes)
        pid_para_txt4 = ax4.text(xi+0.03,yi - 5*dyi,'%2d' % (isteps),fontsize=default_fontsize,va='center',ha='left',transform = ax4.transAxes)

    
        xi = 1.63
        ax4.text(xi,yi,'PID parameters',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
        
        # Mensje de error
        x_error = 1.64
        y_error = 0.01
    
        txt_error = ax4.text(x_error,y_error,'',fontsize=default_fontsize-1,va='center',transform = ax4.transAxes,color='red') 

        self.data_plot1 = data_plot1
        self.data_plot2 = data_plot2
        self.data_plot3 = data_plot3
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
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
        bstop.label.set_size(10)
        
        axonoff = plt.axes([0.88, 0.25, 0.1, 0.075])
        bonoff = Button(axonoff, 'PID ON ')
        bonoff.on_clicked(self.pid_onoff_callback)  
        bonoff.label.set_size(10)
        bonoff.label.set_backgroundcolor([1,1,1,0.8])
        bonoff.label.set_color([0.,0.99,0.])

        axpidk = plt.axes([0.92, 0.88, 0.075, 0.075])
        bpidk = Button(axpidk, 'PID   ')
        bpidk.on_clicked(self.pid_constants_callback)  
        bpidk.label.set_size(10)
        
        
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
        self.axpidk = axpidk
        self.bpidk = bpidk
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
        self.kd = kd
        self.isteps = isteps
        
        
    def pid_onoff_callback(self,event):
        
        pid_onoff_button = self.pid_onoff_button
        bonoff = self.bonoff
        
        if pid_onoff_button is True:
            self.pid_onoff_button = False
            bonoff.label.set_text('PID OFF')
            bonoff.label.set_color([0.99,0.0,0.])
        else:
            self.pid_onoff_button = True  
            bonoff.label.set_text('PID ON ')
            bonoff.label.set_color([0.,0.99,0.])            

    def pid_constants_callback(self,event):
        
        pid_costants_button = self.pid_costants_button
        bpidk = self.bpidk
        
        if pid_costants_button is True:
            self.pid_costants_button = False
            bpidk.label.set_text('PID   ')
        else:
            self.pid_costants_button = True  
            bpidk.label.set_text('PID*K')
             

    def inicializa_semaforos(self):
        self.semaphore1 = threading.Semaphore(0) # Input buffer
        self.semaphore2 = threading.Semaphore(0) # Output buffer
        self.semaphore3 = threading.Semaphore(0) # Guardado de raw data
        self.semaphore4 = threading.Semaphore(0) # Guardado de processed data
        self.semaphore5 = threading.Semaphore(0) # Plot
        

    # Defino el thread que envia la señal          
    def writer_thread(self):  
        
        do_channels_str = self.do_channels_str
        initial_do_frequency = self.initial_do_frequency
        initial_do_duty_cycle = self.initial_duty_cycle
        
        with nidaqmx.Task() as task_do:
            
            task_do.co_channels.add_co_pulse_chan_freq(counter=do_channels_str,duty_cycle=initial_do_duty_cycle,freq=initial_do_frequency,units=nidaqmx.constants.FrequencyUnits.HZ)
            task_do.timing.cfg_implicit_timing(sample_mode=constants.AcquisitionType.CONTINUOUS)    
            digi_s = nidaqmx.stream_writers.CounterWriter(task_do.out_stream)
            task_do.start()

            prev_duty_cycle = 0
            act_duty_cycle = self.initial_do_duty_cycle
                       
            i = 0
            while not self.evento_salida.is_set():
                                
                self.semaphore2.acquire()   
                
                prev_duty_cycle = act_duty_cycle
                act_duty_cycle = self.output_buffer_duty_cycle[i,0]
                
                if act_duty_cycle != prev_duty_cycle:    
                    digi_s.write_one_sample_pulse_frequency(frequency = initial_do_frequency, duty_cycle = act_duty_cycle)
                
                i = i+1
                i = i%buffer_chunks     

        
                       
#        i = 0
#        while not self.evento_salida.is_set():
#            
#            self.semaphore2.acquire()   
#            #time.sleep(0.008)
#            #digi_s.write_one_sample_pulse_frequency(frequency = initial_do_frequency, duty_cycle = output_buffer_duty_cycle[i])
#            
#            i = i+1
#            i = i%buffer_chunks           


               
    # Defino el thread que adquiere la señal   
    def reader_thread(self):
        
        ai_channels_str = self.ai_channels_str
        ai_nbr_channels =  self.ai_nbr_channels
        ai_samples= self.ai_samples
        ai_voltage_range = self.ai_voltage_range
                
        with nidaqmx.Task() as task_ai:
            
            task_ai.ai_channels.add_ai_voltage_chan(ai_channels_str,max_val=ai_voltage_range, min_val=-ai_voltage_range,terminal_config=constants.TerminalConfiguration.RSE) #, "Voltage")#,AIVoltageUnits.Volts)#,max_val=10., min_val=-10.)
            task_ai.timing.cfg_samp_clk_timing(ai_samplerate,samps_per_chan=ai_samples,sample_mode=constants.AcquisitionType.CONTINUOUS)
                
            i = 0
            while not self.evento_salida.is_set():
        
                medicion = task_ai.read(number_of_samples_per_channel=ai_samples)
                medicion = np.asarray(medicion)
                medicion = np.reshape(medicion,ai_nbr_channels*ai_samples,order='F')
                
                for j in range(ai_nbr_channels):
                    self.input_buffer[i,:,j] = medicion[j::ai_nbr_channels]  
                
                self.semaphore1.release() 
                self.semaphore3.release()
                
                i = i+1
                i = i%buffer_chunks                  

#        i = 0
#        dt = self.ai_samples/self.ai_samplerate-0.0005
#        while not self.evento_salida.is_set():
#    
#            medicion = np.zeros([ai_nbr_channels,ai_samples])
#            #medicion[0,:] = np.arange(0,ai_samples)
#            tt = i*ai_samples/ai_samplerate + np.arange(ai_samples)/ai_samples/ai_samplerate
#            medicion[0,:] = 2.1 + 1.0*np.sin(2*np.pi*0.2*tt) + np.random.rand(ai_samples)
#            #medicion[1,:] = 1 + np.random.rand(ai_samples)
#            medicion = np.reshape(medicion,ai_nbr_channels*ai_samples,order='F')
#            
#            for j in range(ai_nbr_channels):
#                self.input_buffer[i,:,j] = medicion[j::ai_nbr_channels]  
#            
#            self.semaphore1.release() 
#            self.semaphore3.release()
#            
#            time.sleep(dt)
#            
#            i = i+1
#            i = i%buffer_chunks  

  
    # Thread del callback        
    def callback_thread(self):
        
        semaforo1_ovr = self.semaforo1_ovr
        semaforo2_ovr = self.semaforo2_ovr
                
        i = 0
        while not self.evento_salida.is_set(): 
            

            if self.semaphore1._value > semaforo1_ovr:
                error_string = 'Hay overrun en llenado del input_buffer!'
                self.exit_callback1(error_string)        
                            
            if self.semaphore2._value > semaforo2_ovr:
                error_string = 'Hay overrun en el vaciado del output_buffer!'
                self.exit_callback1(error_string)            
            
            self.semaphore1.acquire() 
            now = datetime.datetime.now()
            now = now.strftime("%Y-%m-%d %H:%M:%S.%f")
            
#            # Paso anterior del buffer circular
#            j = (i-1)%buffer_chunks
    
            ## Inicio Callback      
            self.output_buffer_mean_data[i,:] = np.mean(self.input_buffer[i,:,:],axis=0)
            self.output_buffer_error_data[i,0] = self.output_buffer_mean_data[i,0] - self.setpoint              
            self.output_buffer_pid_constants[i,:] = np.array([self.setpoint,self.kp,self.ki,self.kd,self.isteps])
         
            # funcion de callback
            output_buffer_duty_cycle_i, output_buffer_error_data_i, termino_p, termino_i, termino_d, setpoint, kp, ki, kd, isteps  = self.callback_pid(self,i)             
            
            # Actualizo los buffers luego del callback
            if self.pid_onoff_button is False:
                output_buffer_duty_cycle_i = initial_do_duty_cycle

            self.output_buffer_duty_cycle[i,0] = output_buffer_duty_cycle_i       
            self.output_buffer_pid_terminos[i,:] = np.array([termino_p, termino_i, termino_d])            
            self.output_buffer_pid_constants[i,:] = np.array([setpoint,kp,ki,kd,isteps])    
            self.output_buffer_error_data[i,0] = output_buffer_error_data_i  
            self.output_buffer_semaforos[i,:] = np.array([self.semaphore1._value, self.semaphore2._value, self.semaphore3._value, self.semaphore4._value, self.semaphore5._value])
            self.output_buffer_timestamp[i] = now
            ## Fin callback
            
            self.semaphore2.release()
            self.semaphore4.release()
            self.semaphore5.release()                                
           
            i = i+1
            i = i%buffer_chunks   
       

    def data_writer1_thread(self):
             
        save_raw_data = self.save_raw_data
        sub_chunk_save = self.sub_chunk_save
        buffer_chunks = self.buffer_chunks
        path_raw_data= self.path_raw_data
        semaforo3_ovr = self.semaforo3_ovr
        
        i = 0
        
        if not save_raw_data:
            while not self.evento_salida.is_set():  
                self.semaphore3.acquire() 
                
        else:
                        
            while not self.evento_salida.is_set():  
    
                if self.semaphore3._value > semaforo3_ovr:
                    error_string = 'Hay overrun en la escritura de datos raw data!'
                    self.exit_callback1(error_string)               
                
                self.semaphore3.acquire() 
                
                if not i%sub_chunk_save: 
                    
                    j = (i-sub_chunk_save)%buffer_chunks  
                    jj = (j+sub_chunk_save-1)%buffer_chunks + 1
                    
                    try:
                        self.save_to_np_file(path_raw_data,self.input_buffer[j:jj,:,:]) 
                    except:
                        warning_string = 'Error: No se guarda raw data'
                        self.warning_callback(warning_string)
               
                i = i+1
                i = i%buffer_chunks   

            
    def data_writer2_thread(self):
        
        save_processed_data= self.save_processed_data
        buffer_chunks = self.buffer_chunks
        sub_chunk_save = self.sub_chunk_save
        path_duty_cycle_data= self.path_duty_cycle_data
        path_mean_data= self.path_mean_data
        path_pid_constants = self.path_pid_constants
        path_pid_terminos = self.path_pid_terminos
        path_semaforos = self.path_semaforos
        path_timestamp = self.path_timestamp
        semaforo4_ovr = self.semaforo4_ovr
                
        i = 0
        
        if not save_processed_data:
            while self.evento_salida.is_set():  
                self.semaphore4.acquire() 
        else:
    
            while not self.evento_salida.is_set(): 
    
                if self.semaphore4._value > semaforo4_ovr:    
                    error_string = 'Hay overrun en la escritura de datos duty cycle!'
                    self.exit_callback1(error_string)                          
                
                self.semaphore4.acquire() 
                
                if not i%sub_chunk_save:
                     
                    j = (i-sub_chunk_save)%buffer_chunks  
                    jj = (j+sub_chunk_save-1)%buffer_chunks + 1 
                    
                    try:
                        self.save_to_np_file(path_duty_cycle_data, self.output_buffer_duty_cycle[j:jj,0]) 
                    except:
                        warning_string = 'Error: No se guarda duty cycle'
                        self.warning_callback(warning_string)                        
    
                    try:
                        self.save_to_np_file(path_mean_data, self.output_buffer_mean_data[j:jj,0]) 
                    except:
                        warning_string = 'Error: No se guarda mean data'
                        self.warning_callback(warning_string)                             
    
                    try:
                        self.save_to_np_file(path_pid_constants, self.output_buffer_pid_constants[j:jj,:]) 
                    except:
                        warning_string = 'Error: No se guardan las constantes pid'
                        self.warning_callback(warning_string)                            
    
                    try:
                        self.save_to_np_file(path_pid_terminos, self.output_buffer_pid_terminos[j:jj,:]) 
                    except:
                        warning_string = 'Error: No se guardan los terminos pid'
                        self.warning_callback(warning_string)                           

                    try:
                        self.save_to_np_file(path_semaforos, self.output_buffer_semaforos[j:jj,:]) 
                    except:
                        warning_string = 'Error: No se guardan los semaforos'
                        self.warning_callback(warning_string)    
                   
                    try:
                        self.save_to_txt_file(path_timestamp, self.output_buffer_timestamp[j:jj]) 
                    except:
                        warning_string = 'Error: No se guardan los timestamp'
                        self.warning_callback(warning_string) 
                                            
                i = i+1
                i = i%buffer_chunks             
        
    
    def plot_thread(self):

        buffer_chunks = self.buffer_chunks
        show_plot = self.show_plot
        sub_chunk_plot = self.sub_chunk_plot
        ai_samples = self.ai_samples
        ai_samplerate = self.ai_samplerate
        semaforo5_ovr = self.semaforo5_ovr
        semaforo4_ovr = self.semaforo4_ovr
        semaforo3_ovr = self.semaforo3_ovr
        semaforo2_ovr = self.semaforo2_ovr
        semaforo1_ovr = self.semaforo1_ovr
         
        # Contador de los semaforos
        x_semaphores = np.zeros(5)
        
        # Para tiempo de medicion y de adquisicion
        previous = datetime.datetime.now()
        delta_t = np.array([])
        delta_t_avg = 10
        
        # Data
        data_plot1 = self.data_plot1
        data_plot2 = self.data_plot2
        data_plot3 = self.data_plot3
        
        line1 = self.line1
        line2 = self.line2
        line3 = self.line3
        setpoint_line = self.setpoint_line
        
           
                
        i = 0
        while not self.evento_salida.is_set(): 
        
            if self.semaphore5._value > semaforo5_ovr:
                    error_string = 'Hay overrun en el plot!'
                    self.exit_callback1(error_string)         


            if self.evento_warning.is_set():                
                self.print_error(self.warnings)
                self.warnings = []
                self.evento_warning.clear() 

            
            self.semaphore5.acquire()
                    
            
            if not show_plot:
                continue
            
            if i%sub_chunk_plot == 0: 
                
                # Paso para buffer
                j = (i-sub_chunk_plot)%buffer_chunks  
                jj = (j+sub_chunk_plot-1)%buffer_chunks + 1 
    
                # Medición de tiempos
                now, previous, delta_t, measure_adq_ratio = self.measure_adq_ratio_function(previous,delta_t,delta_t_avg,ai_samples,ai_samplerate,sub_chunk_plot)
                self.txt_now.set_text(now)
            
                # Data update
                self.update_plot(data_plot1,line1,self.output_buffer_mean_data,sub_chunk_plot,j,jj)
                self.update_plot(data_plot2,line2,self.output_buffer_duty_cycle,sub_chunk_plot,j,jj)
                if not self.pid_costants_button:
                    self.update_plot(data_plot3,line3,self.output_buffer_pid_terminos,sub_chunk_plot,j,jj)
                else:
                    self.update_plot(data_plot3,line3,self.output_buffer_pid_terminos*self.output_buffer_pid_constants[:,1:4],sub_chunk_plot,j,jj)
                setpoint_line.set_ydata(self.output_buffer_pid_constants[i,0])
 
                # Textos
                x_semaphores[0] = (self.semaphore1._value/semaforo1_ovr)*100
                x_semaphores[1] = (self.semaphore2._value/semaforo2_ovr)*100
                x_semaphores[2] = (self.semaphore3._value/semaforo3_ovr)*100
                x_semaphores[3] = (self.semaphore4._value/semaforo4_ovr)*100
                x_semaphores[4] = (self.semaphore5._value/semaforo5_ovr)*100
                
                self.monitoring_txt0.set_text('%2d' % x_semaphores[0] + ' %')
                self.monitoring_txt1.set_text('%2d' % x_semaphores[1] + ' %')
                self.monitoring_txt2.set_text('%2d' % x_semaphores[2] + ' %')
                self.monitoring_txt3.set_text('%2d' % x_semaphores[3] + ' %')
                self.monitoring_txt4.set_text('%2d' % x_semaphores[4] + ' %')              
                self.monitoring_txt5.set_text('%4.2f' % measure_adq_ratio)
#               
                self.pid_para_txt0.set_text('%6.2f' % self.output_buffer_pid_constants[i,0] + ' V')
                self.pid_para_txt1.set_text('%6.2f' % self.output_buffer_pid_constants[i,1])
                self.pid_para_txt2.set_text('%6.2f' % self.output_buffer_pid_constants[i,2])
                self.pid_para_txt3.set_text('%6.2f' % self.output_buffer_pid_constants[i,3])
                self.pid_para_txt4.set_text('%2d' % self.output_buffer_pid_constants[i,4])                                
                
            self.fig.canvas.draw_idle()
                   
            i = i+1
            i = i%buffer_chunks 
            
        self.print_error(self.acquiring_errors)
        self.fig.canvas.draw_idle()


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

    def exit_callback(self,event):  
        self.acquiring_errors.append('Medición interrumpida por el usuario')
        self.evento_salida.set()

    def save_to_np_file(self,filename,arr):
        f_handle = open(filename, 'ab')
        np.save(f_handle, arr)
        f_handle.close()    
    
    def save_to_txt_file(self, filename,arr):
        with open(filename,'a') as f:    
            for item in arr:
                f.write('%s \n' % item)        


    def save_parametros_to_txt_file(self, filename):
        with open(filename,'w') as f:  
            now = datetime.datetime.now()
            now = now.strftime("%Y-%m-%d %H:%M:%S.%f")  
            
            f.write('%s : %s\n' % ('fecha_de_inicio',now))    
            f.write('%s : %d\n' % ('buffer_chunks',self.buffer_chunks))                          
            f.write('%s : %d\n' % ('ai_samples',self.ai_samples))         
            f.write('%s : %d\n' % ('ai_samplerate',self.ai_samplerate))       
            f.write('%s : %s\n' % ('ai_device',self.ai_device))     
            f.write('%s : %f\n' % ('ai_voltage_range',self.ai_voltage_range))  
            f.write('%s : %s\n' % ('do_device',self.do_device))   
            f.write('%s : %f\n' % ('initial_do_duty_cycle',self.initial_do_duty_cycle))  
            f.write('%s : %d\n' % ('initial_do_frequency',self.initial_do_frequency))  
            f.write('%s : %d\n' % ('save_raw_data',self.save_raw_data)) 
            f.write('%s : %d\n' % ('save_processed_data',self.save_processed_data)) 
            f.write('%s : %s\n' % ('path_data_save',self.path_data_save)) 
            f.write('%s : %d\n' % ('show_plot',self.show_plot)) 
            f.write('%s : %d\n' % ('sub_chunk_save',self.sub_chunk_save)) 
            f.write('%s : %d\n' % ('sub_chunk_plot',self.sub_chunk_plot)) 
            f.write('%s : %d\n' % ('nbr_buffers_plot',self.nbr_buffers_plot)) 
            f.write('%s : %d,%d,%d,%d,%d \n' % ('semaforos_ovr',self.semaforo1_ovr,self.semaforo2_ovr,self.semaforo3_ovr,self.semaforo4_ovr,self.semaforo5_ovr)) 


    def measure_adq_ratio_function(self,previous,delta_t,delta_t_avg,ai_samples,ai_samplerate,sub_chunk_plot):
        # Medición de tiempos
        now = datetime.datetime.now()
        delta_ti = now - previous
        previous = now
        now = now.strftime("%Y-%m-%d %H:%M:%S")
        
        delta_ti = delta_ti.total_seconds()
        delta_t = np.append(delta_t,delta_ti)
        if delta_t.shape[0] > delta_t_avg:
            delta_t = delta_t[-delta_t_avg:]
            
        measure_adq_ratio = 0
        if delta_t.mean() > 0:
            measure_adq_ratio = (ai_samples/ai_samplerate*sub_chunk_plot)/delta_t.mean()
            
        return now, previous, delta_t, measure_adq_ratio
    
    def load_from_np_file(self,filename):
    
        f = open(filename, 'rb')
        arr = np.load(f)  
        while True:
            try:
                arr = np.append(arr,np.load(f),axis=0)
            except:
                break
        f.close()  
    
        return arr   

    def configura_duty_cycle(self,
        min_duty_cycle,
        max_duty_cycle
    ):
        
        self.min_duty_cycle = min_duty_cycle
        self.max_duty_cycle = max_duty_cycle


    def define_limites_de_plots(self,
        ax1_range=[np.NaN,np.NaN],
        ax2_range=[np.NaN,np.NaN],
        ax3_range=[np.NaN,np.NaN]
    ):
        
        ax1_lim_i = ax1_range[0]
        ax1_lim_f = ax1_range[1]
        ax2_lim_i = ax2_range[0]
        ax2_lim_f = ax2_range[1]        
        ax3_lim_i = ax3_range[0]
        ax3_lim_f = ax1_range[1]
        
        if ax1_lim_i is not np.NaN and ax1_lim_f is not np.NaN:
            if not 'ax1' in self.__dict__:
                self.ax1_lim_i = ax1_lim_i
                self.ax1_lim_f = ax1_lim_f
            else:
                self.ax1.set_ylim([ax1_lim_i,ax1_lim_f])
                
        if ax2_lim_i is not np.NaN and ax2_lim_f is not np.NaN:    
            if not 'ax' in self.__dict__:
                self.ax2_lim_i = ax2_lim_i
                self.ax2_lim_f = ax2_lim_f
            else:
                self.ax2.set_ylim([ax2_lim_i,ax2_lim_f])
                
        if ax3_lim_i is not np.NaN and ax3_lim_f is not np.NaN:       
            if not 'ax3' in self.__dict__:
                self.ax3_lim_i = ax3_lim_i
                self.ax3_lim_f = ax3_lim_f      
            else:
                self.ax3.set_ylim([ax3_lim_i,ax3_lim_f])
            

    def define_limites_sliders(self,
        setpoint_range = [np.NaN,np.NaN],
        kp_range = [np.NaN,np.NaN],
        ki_range = [np.NaN,np.NaN],
        kd_range = [np.NaN,np.NaN],
    ):
        
        setpoint_min = setpoint_range[0]
        setpoint_max = setpoint_range[1]
        kp_min = kp_range[0]
        kp_max = kp_range[1]
        ki_min = ki_range[0]
        ki_max = ki_range[1]
        kd_min = kd_range[0]
        kd_max = kd_range[1]
        
        if setpoint_min is not np.NaN and setpoint_max is not np.NaN:
            if not 'ssetpoint' in self.__dict__: 
                self.setpoint_min = setpoint_min
                self.setpoint_max = setpoint_max            
            else:  
                self.ssetpoint.valmin = setpoint_min
                self.ssetpoint.valmax = setpoint_max
                self.ssetpoint.ax.set_xlim(setpoint_min,setpoint_max) 

        if kp_min is not np.NaN and kp_max is not np.NaN:
            if not 'skp' in self.__dict__:  
                self.kp_min = kp_min
                self.kp_max = kp_max            
            else:  
                self.skp.valmin = kp_min
                self.skp.valmax = kp_max
                self.skp.ax.set_xlim(kp_min,kp_max) 

        if ki_min is not np.NaN and ki_max is not np.NaN:
            if not 'ski' in self.__dict__:   
                self.ki_min = ki_min
                self.ki_max = ki_max            
            else:  
                self.ski.valmin = ki_min
                self.ski.valmax = ki_max
                self.ski.ax.set_xlim(ki_min,ki_max) 
        
        if kd_min is not np.NaN and kd_max is not np.NaN:
            if not 'skd' in self.__dict__:  
                self.kd_min = kd_min
                self.kd_max = kd_max            
            else:  
                self.skd.valmin = kd_min
                self.skd.valmax = kd_max
                self.skd.ax.set_xlim(kd_min,kd_max) 


    def define_semaforos_overrun(self,
        semaforo1_ovr = np.NaN,
        semaforo2_ovr = np.NaN,
        semaforo3_ovr = np.NaN,
        semaforo4_ovr = np.NaN,
        semaforo5_ovr = np.NaN
    ):
      
        if semaforo1_ovr > 0 and semaforo1_ovr < self.buffer_chunks and semaforo1_ovr is not np.NaN:        
            self.semaforo1_ovr = semaforo1_ovr
        if semaforo2_ovr > 0 and semaforo2_ovr < self.buffer_chunks and semaforo2_ovr is not np.NaN:              
            self.semaforo2_ovr = semaforo2_ovr
        if semaforo3_ovr > 0 and semaforo3_ovr < self.buffer_chunks and semaforo3_ovr is not np.NaN:             
            self.semaforo3_ovr = semaforo3_ovr
        if semaforo4_ovr > 0 and semaforo4_ovr < self.buffer_chunks and semaforo4_ovr is not np.NaN:              
            self.semaforo4_ovr = semaforo4_ovr            
        if semaforo5_ovr > 0 and semaforo5_ovr < self.buffer_chunks and semaforo5_ovr is not np.NaN:             
            self.semaforo5_ovr = semaforo5_ovr
            
        

    def update_plot(self,data_plot,line_plot,output_buffer,sub_chunk_plot,j,jj):
        
            data_plot[0:-sub_chunk_plot,:] = data_plot[sub_chunk_plot:,:]
            data_plot[-sub_chunk_plot:,:] = output_buffer[j:jj,:]
                                                
            for k in range(data_plot.shape[1]): 
                line_plot[k].set_ydata(data_plot[:,k])  
                

    def inicializa_threads(self):    
        self.t1 = threading.Thread(target=self.writer_thread, args=[])
        self.t2 = threading.Thread(target=self.reader_thread, args=[])
        self.t3 = threading.Thread(target=self.callback_thread, args=[])
        self.t4 = threading.Thread(target=self.data_writer1_thread, args=[])
        self.t5 = threading.Thread(target=self.data_writer2_thread, args=[])
        self.t6 = threading.Thread(target=self.plot_thread, args=[])

    

    def configura_adquisicion(self):
        self.inicializa_errores()
        self.inicializa_variables()
        self.acondiciona_variables()
        self.inicializa_buffers()        
        self.inicializa_interfaz()    
        self.inicializa_sliders_botones()
        self.inicializa_semaforos()
        self.inicializa_threads()
        
        if len(self.warnings) > 0:          
            self.print_error(self.warnings)
            
        if len(self.initialize_errors) > 0:
            self.initialize_errors = [u'Error de inicialización'] + self.initialize_errors
            self.print_error(self.initialize_errors)
                    

    def start(self):
        
        if self.save_raw_data or self.save_processed_data:
            self.save_parametros_to_txt_file(self.path_parametros)            
        
        if len(self.initialize_errors) == 0:
            self.t1.start()
            self.t2.start()
            self.t3.start()
            self.t4.start()
            self.t5.start()
            self.t6.start()
        

#%%
        
def callback_pid(self,i):

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
    max_duty_cycle = self.max_duty_cycle
    min_duty_cycle = self.min_duty_cycle 
    
    # Cargo valores actuales
    setpoint = self.output_buffer_pid_constants[i,0]
    kp = self.output_buffer_pid_constants[i,1]
    ki = self.output_buffer_pid_constants[i,2]
    kd = self.output_buffer_pid_constants[i,3]
    isteps = int(self.output_buffer_pid_constants[i,4])
    output_buffer_error_data_i = self.output_buffer_error_data[i,0]
    mean_data_i = self.output_buffer_mean_data[i,0]
    sample_period = self.sample_period
        
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
    termino_d = output_buffer_error_data_i - self.output_buffer_error_data[j]
    termino_d = termino_d*sample_period
    
    # Termino integral (hay que optimizar esto)
    termino_i = 0
    if k >= i:
        termino_i = np.sum(self.output_buffer_error_data[k:buffer_chunks]) + np.sum(self.output_buffer_error_data[0:i])
    else:
        termino_i = np.sum(self.output_buffer_error_data[k:i])
    termino_i = termino_i*sample_period
    
    output_buffer_duty_cycle_i =  kp*termino_p + ki*termino_i + kd*termino_d
    
    #time.sleep(0.01)

    # Salida de la función
    output_buffer_duty_cycle_i = min(output_buffer_duty_cycle_i,max_duty_cycle)
    output_buffer_duty_cycle_i = max(output_buffer_duty_cycle_i,min_duty_cycle)
    
    return output_buffer_duty_cycle_i, output_buffer_error_data_i, termino_p, termino_i, termino_d, setpoint, kp, ki, kd, isteps 



##
carpeta_salida = 'PID3'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
         
# Variables 
ai_channels = [4]
buffer_chunks = 500
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
path_data_save = os.path.join(carpeta_salida,'experimento')
callback_pid_variables = {}



##
parametros = {}
parametros['buffer_chunks'] = buffer_chunks
parametros['ai_samples'] = ai_samples
parametros['ai_samplerate'] = ai_samplerate
parametros['ai_device'] = 'Dev2/ai'
parametros['ai_channels'] = ai_channels  
parametros['ai_voltage_range'] = 5
parametros['do_device'] = 'Dev2/ctr'
parametros['do_channel'] = do_channel  
parametros['initial_do_duty_cycle'] = initial_do_duty_cycle
parametros['initial_do_frequency'] = initial_do_frequency
parametros['setpoint'] = setpoint
parametros['pid_constants'] = [kp,ki,kd,isteps]
parametros['save_raw_data'] = True
parametros['save_processed_data'] = True
parametros['path_data_save'] = path_data_save
parametros['show_plot'] = True
parametros['callback_pid'] = callback_pid    
parametros['callback_pid_variables'] = callback_pid_variables
parametros['sub_chunk_save'] = 25
parametros['sub_chunk_plot'] = 25
parametros['nbr_buffers_plot'] = 10
parametros['plot_rate_hz'] = 10

adquisicion = pid_daq(parametros)
adquisicion.configura_adquisicion()
adquisicion.define_limites_de_plots(ax1_range=[0,5],ax2_range=[0,2],ax3_range=[-5,5])
adquisicion.define_limites_sliders(setpoint_range=[0,5],kp_range=[0,50],ki_range=[0,50],kd_range=[0,25])
adquisicion.define_semaforos_overrun(semaforo1_ovr = 5)
adquisicion.start()        
        
