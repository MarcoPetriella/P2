
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
import serial
import struct

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
        self.com_device = parametros['com_device']
        self.setpoint = parametros['setpoint']
        self.initial_value = parametros['initial_value']
        self.save_data = parametros['save_data']
        self.path_data_save = parametros['path_data_save']  
        parametros_pid = parametros['pid_constants']
        self.kp = parametros_pid[0]
        self.ki = parametros_pid[1]
        self.kd = parametros_pid[2]
        self.isteps = parametros_pid[3]  
        self.dt = parametros['dt']
        self.chunk_plot = parametros['chunk_plot']
        
        self.cantidad_variables = 13
        
        # Valores predeterminado
        self.pid_onoff_button = True
        self.pid_costants_button = False
        self.plot_fontsize = 10
        self.default_fontsize = 6

        self.setpoint_min = 0.550
        self.setpoint_max = 2.750
        self.kp_min = 0
        self.kp_max = 10
        self.ki_min = 0
        self.ki_max = 60
        self.kd_min = 0
        self.kd_max = 60
        
        self.ax1_lim_i = 0
        self.ax1_lim_f = 5
        self.ax2_lim_i = 0
        self.ax2_lim_f = 3.3
        self.ax3_lim_i = -2
        self.ax3_lim_f = 2 
        
        # Overrun de semaforos
        buffer_chunks = self.buffer_chunks
        self.semaforo1_ovr = buffer_chunks
        self.semaforo2_ovr = buffer_chunks
        self.semaforo3_ovr = buffer_chunks
        self.semaforo4_ovr = buffer_chunks
        self.semaforo5_ovr = buffer_chunks
        
                     
        
    def acondiciona_variables(self):
        # Archivos de salida
        self.path_processed_data_save = self.path_data_save + '_valores.bin'
        self.path_timestamp = self.path_data_save + '_timestamp.txt'
        self.path_parametros = self.path_data_save + '_parametros_iniciales.txt'
        
        if self.path_processed_data_save:
            if os.path.exists(self.path_processed_data_save):
                os.remove(self.path_processed_data_save)
                
        if self.path_timestamp:
            if os.path.exists(self.path_timestamp):
                os.remove(self.path_timestamp)
                
        if self.path_parametros:
            if os.path.exists(self.path_parametros):
                os.remove(self.path_parametros)                
        
        parametros = self.parametros
        buffer_chunks = self.buffer_chunks
        sub_chunk_save = np.NaN
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
            


    
        # pasos de la integral del pid
        paso_integral = 10
        possible_isteps = np.arange(0,buffer_chunks+paso_integral,paso_integral,dtype=int)
        possible_isteps[0] = 1
    
        # Largo del vector a graficar
        if 'nbr_buffers_plot' in parametros:
            nbr_buffers_plot = int(parametros['nbr_buffers_plot'])
        else:
            nbr_buffers_plot = 10
            
        
        self.sub_chunk_save = sub_chunk_save
        self.possible_isteps = possible_isteps
        self.nbr_buffers_plot = nbr_buffers_plot
        self.initialize_errors = initialize_errors
        self.warnings = warnings

    def inicializa_buffers(self):
    
        buffer_chunks = self.buffer_chunks
        cantidad_variables = self.cantidad_variables
         
        input_buffer = np.zeros([buffer_chunks,cantidad_variables],dtype=np.float)
        input_buffer_timestamp = buffer_chunks*[None]
        
        self.input_buffer = input_buffer
        self.input_buffer_timestamp = input_buffer_timestamp


 
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
        chunk_plot = self.chunk_plot
        dt = self.dt
        
        setpoint = self.setpoint
        kp =self.kp
        ki = self.ki
        kd = self.kd
        isteps = self.isteps        

        data_plot1 = np.zeros([buffer_chunks*nbr_buffers_plot,2])
        data_plot2 = np.zeros([buffer_chunks*nbr_buffers_plot,1])    
        data_plot3 = np.zeros([buffer_chunks*nbr_buffers_plot,3])  
        
        tiempo_vec = np.arange(0,data_plot1.shape[0])*chunk_plot*dt
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
        ax2.set_ylabel('control [V]',fontsize = plot_fontsize)
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
        
#        ax4.text(xi,yi,'Pending processes',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
#        ax4.text(xi,yi - 1*dyi,'Input buffer filling: ',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
#        ax4.text(xi,yi - 2*dyi,'Output buffer emptying:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
#        ax4.text(xi,yi - 3*dyi,'Raw data writer:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
#        ax4.text(xi,yi - 4*dyi,'Processed data writer:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
#        ax4.text(xi,yi - 5*dyi,'Plotting:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
#        ax4.text(xi,yi - 6*dyi,'Measuring acquiring ratio:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
#        
#        xi = 1.53
#        monitoring_txt0 = ax4.text(xi,yi - 1*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
#        monitoring_txt1 = ax4.text(xi,yi - 2*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
#        monitoring_txt2 = ax4.text(xi,yi - 3*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
#        monitoring_txt3 = ax4.text(xi,yi - 4*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
#        monitoring_txt4 = ax4.text(xi,yi - 5*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
#        monitoring_txt5 = ax4.text(xi,yi - 6*dyi,str(0) + '%',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
#        
#        xi = -0.40
#        ax4.text(xi,yi,'Acquisition parameters',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
#        ax4.text(xi,yi - 1*dyi,'Samplerate: ',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
#        ax4.text(xi,yi - 2*dyi,'Samples per chunk:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
#        ax4.text(xi,yi - 3*dyi,'Nbr. chunks:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
#        ax4.text(xi,yi - 4*dyi,'PWM frequency:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
#        ax4.text(xi,yi - 5*dyi,'Nbr. PWM cycles per chunk:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
#        ax4.text(xi,yi - 6*dyi,'Save raw / processed data:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
#        ax4.text(xi,yi - 7*dyi,'Display plot chunks / rate:',fontsize=default_fontsize,va='center',transform = ax4.transAxes)
#        
#        xi = 0.55
#        ax4.text(xi,yi - 1*dyi,'%6.2f' % (ai_samplerate/1000.0) + ' kHz',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
#        ax4.text(xi,yi - 2*dyi,'%4d' % ai_samples + ' / ' + '%6.2f' % (ai_samples/ai_samplerate*1000.) + ' ms',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
#        ax4.text(xi,yi - 3*dyi,'%4d' % buffer_chunks + ' / ' + '%6.2f' % (buffer_chunks*ai_samples/ai_samplerate) + ' s',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
#        ax4.text(xi,yi - 4*dyi,'%6.2f' % (initial_do_frequency/1000.) + ' kHz',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
#        ax4.text(xi,yi - 5*dyi,'%6.2f' % ((ai_samples/ai_samplerate)*(initial_do_frequency)) ,fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
#        ax4.text(xi,yi - 6*dyi, str(save_raw_data) + ' / ' + str(save_processed_data),fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
#        ax4.text(xi,yi - 7*dyi, '%4d' % sub_chunk_plot + ' / ' +'%6.2f' % (ai_samplerate/ai_samples/sub_chunk_plot) + ' Hz',fontsize=default_fontsize,va='center',ha='right',transform = ax4.transAxes)
    
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
        self.semaphore1 = threading.Semaphore(0) # plot
        self.semaphore2 = threading.Semaphore(0) # guardado
        
        
    def inicializa_arduino(self):
        self.arduino = serial.Serial('COM'+str(self.com_device), 2*9600, timeout=1)
        self.arduino.set_buffer_size(rx_size = 8*1000, tx_size = 8*1000)
        
                       
    def recibe(self):
        
        cantidad_variables = self.cantidad_variables
        buffer_chunks = self.buffer_chunks
        
        i = 0
        while not self.evento_salida.is_set():  
            
            try:            
                rawString = self.arduino.read(4*cantidad_variables)
                array_serial = struct.unpack(cantidad_variables*'f',rawString)
                                
                now = datetime.datetime.now()
                now = now.strftime("%Y-%m-%d %H:%M:%S.%f")  
                
                
                if len(array_serial) == cantidad_variables:
                    self.input_buffer[i,:] = array_serial
                    self.input_buffer_timestamp[i] = now
                    i = i+1
                    i = i%buffer_chunks
                    self.semaphore1.release()
                    self.semaphore2.release()
                    print(array_serial)
                
            except:
                print("Error en la lectura")
            
            while self.arduino.inWaiting() > 200:
                print('Limpiando buffer')
                self.arduino.read(4)
                    

    def manda(self):   
     
        self.arduino.flush()
        while not self.evento_salida.is_set():     
            try:     
#                self.arduino.flush()
#                time.sleep(1)
                
                #self.arduino.write(struct.pack('<fffffff',self.setpoint,self.kp,self.ki,self.kd,self.isteps,float(self.pid_onoff_button),self.initial_value)) 
                self.arduino.flushOutput()
                time.sleep(1)   
                self.arduino.write(struct.pack('<fffff',self.setpoint,self.kp,self.ki,self.kd,self.isteps)) 
                time.sleep(1)   

                
            except:
                print("Error en el envio")


    def guarda(self):
             

        sub_chunk_save = self.sub_chunk_save
        buffer_chunks = self.buffer_chunks

        
        path_processed_data_save = self.path_processed_data_save
        path_timestamp =  self.path_timestamp
        
        i = 0
        
        if not self.save_data:
            while not self.evento_salida.is_set():  
                self.semaphore2.acquire() 
                
        else:
                        
            while not self.evento_salida.is_set():  
    
                if self.semaphore2._value > buffer_chunks:
                    error_string = 'Hay overrun en la escritura de datos raw data!'
                    self.exit_callback1(error_string)               
                
                self.semaphore2.acquire() 
                
                if not i%sub_chunk_save: 
                    
                    j = (i-sub_chunk_save)%buffer_chunks  
                    jj = (j+sub_chunk_save-1)%buffer_chunks + 1
                    
                    try:
                        self.save_to_np_file(path_processed_data_save,self.input_buffer[j:jj,:]) 
                    except:
                        warning_string = 'Error: No se guarda raw data'
                        self.warning_callback(warning_string)
                        
                        
                    try:
                        self.save_to_txt_file(path_timestamp, self.input_buffer_timestamp[j:jj]) 
                    except:
                        warning_string = 'Error: No se guardan los timestamp'
                        self.warning_callback(warning_string)                         
                        
               
                i = i+1
                i = i%buffer_chunks   

                       
        
    
    def plotea(self):
        
        buffer_chunks = self.buffer_chunks
        
        # Data
        data_plot1 = self.data_plot1
        data_plot2 = self.data_plot2
        data_plot3 = self.data_plot3
        
        line1 = self.line1
        line2 = self.line2
        line3 = self.line3
        setpoint_line = self.setpoint_line
        
        sub_chunk_plot = 1
                        
        i = 0
        while not self.evento_salida.is_set(): 
        
            if self.semaphore1._value > buffer_chunks:
                    error_string = 'Hay overrun en el plot!'
                    self.exit_callback1(error_string)         


            if self.evento_warning.is_set():                
                self.print_error(self.warnings)
                self.warnings = []
                self.evento_warning.clear() 

            self.semaphore1.acquire()  
            
            if not i%sub_chunk_plot: 
                
                # Paso para buffer
                j = (i-sub_chunk_plot)%buffer_chunks  
                jj = (j+sub_chunk_plot-1)%buffer_chunks + 1 
                
                now = datetime.datetime.now()
                now = now.strftime("%Y-%m-%d %H:%M:%S")                  
                self.txt_now.set_text(now)
    
            
                # Data update
                self.update_plot(data_plot1,line1,self.input_buffer[:,8:10],sub_chunk_plot,j,jj)
                self.update_plot(data_plot2,line2,self.input_buffer[:,10:11],sub_chunk_plot,j,jj)
                if not self.pid_costants_button:
                    self.update_plot(data_plot3,line3,self.input_buffer[:,5:8],sub_chunk_plot,j,jj)
                else:
                    self.update_plot(data_plot3,line3,self.input_buffer[:,5:8]*self.input_buffer[:,1:4],sub_chunk_plot,j,jj)
                setpoint_line.set_ydata(self.input_buffer[i,0])

            
                self.pid_para_txt0.set_text('%6.2f' % self.input_buffer[i,0] + ' V')
                self.pid_para_txt1.set_text('%6.2f' % self.input_buffer[i,1])
                self.pid_para_txt2.set_text('%6.2f' % self.input_buffer[i,2])
                self.pid_para_txt3.set_text('%6.2f' % self.input_buffer[i,3])
                self.pid_para_txt4.set_text('%2d' % self.input_buffer[i,4])                                
                
            self.fig.canvas.draw_idle()
                   
            i = i+1
            i = i%buffer_chunks 
            
        self.arduino.close()    
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
            if not 'ax2' in self.__dict__:
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
        semaforo2_ovr = np.NaN
    ):
      
        if semaforo1_ovr > 0 and semaforo1_ovr < self.buffer_chunks and semaforo1_ovr is not np.NaN:        
            self.semaforo1_ovr = semaforo1_ovr
        if semaforo2_ovr > 0 and semaforo2_ovr < self.buffer_chunks and semaforo2_ovr is not np.NaN:              
            self.semaforo2_ovr = semaforo2_ovr
            
        

    def update_plot(self,data_plot,line_plot,output_buffer,sub_chunk_plot,j,jj):
        
            data_plot[0:-sub_chunk_plot,:] = data_plot[sub_chunk_plot:,:]
            data_plot[-sub_chunk_plot:,:] = output_buffer[j:jj,:]
                                                
            for k in range(data_plot.shape[1]): 
                line_plot[k].set_ydata(data_plot[:,k])  
                

    def inicializa_threads(self):    
        self.t1 = threading.Thread(target=self.recibe, args=[])
        self.t2 = threading.Thread(target=self.guarda, args=[])
        self.t3 = threading.Thread(target=self.manda, args=[])
        self.t4 = threading.Thread(target=self.plotea, args=[])


    

    def configura_adquisicion(self):
        self.inicializa_errores()
        self.inicializa_variables()
        self.acondiciona_variables()
        self.inicializa_buffers()        
        self.inicializa_interfaz()    
        self.inicializa_sliders_botones()
        self.inicializa_arduino()
        self.inicializa_semaforos()
        self.inicializa_threads()
        
        if len(self.warnings) > 0:          
            self.print_error(self.warnings)
            
        if len(self.initialize_errors) > 0:
            self.initialize_errors = [u'Error de inicialización'] + self.initialize_errors
            self.print_error(self.initialize_errors)
                    

    def start(self):
        
#        if self.save_data:
#            self.save_parametros_to_txt_file(self.path_parametros)            
        
        if len(self.initialize_errors) == 0:
            self.t1.start()
            self.t2.start()
            self.t3.start()
            self.t4.start()

        

#%%
        


##
carpeta_salida = 'PID_ARDUINO'

if not os.path.exists(carpeta_salida):
    os.mkdir(carpeta_salida)
         
# Variables 
buffer_chunks = 500
initial_value = 1.5
setpoint = 2.3
kp = 0.96
ki = 22.98
kd = 19.86
isteps = 60
path_data_save = os.path.join(carpeta_salida,'experimento6')
callback_pid_variables = {}


##
parametros = {}
parametros['com_device'] = 5
parametros['buffer_chunks'] = buffer_chunks
parametros['initial_value'] = initial_value
parametros['setpoint'] = setpoint
parametros['pid_constants'] = [kp,ki,kd,isteps]
parametros['save_data'] = True
parametros['path_data_save'] = path_data_save
parametros['sub_chunk_save'] = 25
parametros['nbr_buffers_plot'] = 1
parametros['dt'] = 1./400.
parametros['chunk_plot'] = 20


adquisicion = pid_daq(parametros)
adquisicion.configura_adquisicion()
adquisicion.define_limites_de_plots(ax1_range=[0,3.5],ax2_range=[0,3.5],ax3_range=[-2,2])
adquisicion.define_limites_sliders(setpoint_range=[0.535,2.750],kp_range=[0,2],ki_range=[0,50],kd_range=[0,50])
adquisicion.start()        
        
