# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 12:36:24 2018

@author: Marco
"""



#%%
#import matplotlib
#matplotlib.use('GTKAgg') 
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

#from pyqtgraph.Qt import QtGui, QtCore
#import pyqtgraph as pg

import nidaqmx
import nidaqmx.constants as constants
import nidaqmx.stream_writers
    
#params = {'legend.fontsize': 'medium',
#     #     'figure.figsize': (15, 5),
#         'axes.labelsize': 'medium',
#         'axes.titlesize':'medium',
#         'xtick.labelsize':'medium',
#         'ytick.labelsize':'medium'}
#pylab.rcParams.update(params)


def cross_correlation_using_fft(x, y):
    """
    Se utiliza el algoritmo propuesto en:
        https://lexfridman.com/fast-cross-correlation-and-time-series-synchronization-in-python/
    
    """
    
    f1 = fft.fft(x)
    f2 = fft.fft(np.flipud(y))
    cc = np.real(fft.ifft(f1 * f2))
    return fft.fftshift(cc)


def barra_progreso(paso,pasos_totales,leyenda,tiempo_ini):

    """
    Esta función genera una barra de progreso
    
    Parametros:
    -----------
    paso : int, paso actual
    pasos_totales : int, cantidad de pasos totales
    leyenda : string, leyenda de la barra
    tiempo_ini : datetime, tiempo inicial de la barra en formato datetime   
    
    Autores: Leslie Cusato, Marco Petriella
    """    
    n_caracteres = 30    
    barrita = int(n_caracteres*(paso+1)/pasos_totales)*chr(9632)    
    ahora = datetime.datetime.now()
    eta = (ahora-tiempo_ini).total_seconds()
    eta = (pasos_totales - paso - 1)/(paso+1)*eta
       
    stdout.write("\r %s %s %3d %s |%s| %s %6.1f %s" % (leyenda,':',  int(100*(paso+1)/pasos_totales), '%', barrita.ljust(n_caracteres),'ETA:',eta,'seg'))
    
    if paso == pasos_totales-1:
        print("\n")    
    
    

def signalgen(type,fr,amp,duration,fs):
    """
    generates different signals with len(duration*fs)
    type: 'sin', 'square', 'ramp', 'constant'
    fr: float, frequency of the signal in Hz
    amp: float, amplitud of the signal
    duration: float, duration of the signal in s
    fs: float, sampling rate in Hz
    output: array, signal generated
    """
    # output=np.array[()]
    if type == 'sine':
        output = amp*np.sin(2.*np.pi*np.arange(int(duration*fs))*fr/fs)
    elif type == 'square':
        output = amp*signal.square(2.*np.pi*np.arange(int(duration*fs))*fr/fs)
    elif type == 'ramp':
        output = amp*signal.sawtooth(2*np.pi*np.arange(int(duration*fs))*fr/fs, width=0.5).astype(np.float32)                 
    elif type == 'constant':
        output = np.full(len(input),amp)
    else:
        print ('wrong signal type')
        output = 0
    return output


def signalgen_corrected(type,fr,amp,duration,fs,frec_fft,power_fft,frec_range):
    """
    generates different signals with len(duration*fs)
    type: 'sin', 'square', 'ramp', 'constant'
    fr: float, frequency of the signal in Hz
    amp: float, amplitud of the signal
    duration: float, duration of the signal in s
    fs: float, sampling rate in Hz
    output: array, signal generated
    """
    
    output = signalgen(type,fr,amp,duration,fs)
    
    # Corrección por respuesta del emisor receptor
    power_fft = np.append(power_fft,power_fft[:0:-1])
    frec_fft = np.append(frec_fft,frec_fft[1::]+frec_fft[len(frec_fft)-1])    
    
    fft_output = fft.fft(output)
    frec_output = np.linspace(0,fft_output.shape[0]-1,fft_output.shape[0])/(fft_output.shape[0]-1)*fs
    
    fft_power_interp = np.interp(frec_output, frec_fft, power_fft)   
    fft_output = fft_output/np.sqrt(fft_power_interp)
    
    ind_frec0 = np.argmin(np.abs(frec_output-frec_range[0]))
    ind_frec1 = np.argmin(np.abs(frec_output-frec_range[1]))
    
    max_fft_output = np.max(np.abs(fft_output[ind_frec0:ind_frec1]))
    fft_output[np.abs(fft_output) > max_fft_output] = fft_output[np.abs(fft_output) > max_fft_output]/np.abs(fft_output[np.abs(fft_output) > max_fft_output])*max_fft_output
    
    output = np.real(fft.ifft(fft_output))
    output = amp*output/np.max(output)
    
    return output
    


def play_rec(fs,input_channels,data_out,corrige_retardos,offset_correlacion=0,steps_correlacion=0,delay=0.0,dato='int32'):
    
    
    """
    Descripción:
    ------------
    Esta función permite utilizar la placa de audio de la pc como un generador de funciones / osciloscopio
    con dos canales de entrada y dos de salida. Para ello utiliza la libreria pyaudio y las opciones de write() y read()
    para escribir y leer los buffer de escritura y lectura. Para realizar el envio y adquisición simultánea de señales, utiliza
    un esquema de tipo productor-consumidor que se ejecutan en thread o hilos diferenntes. Para realizar la comunicación 
    entre threads y evitar overrun o sobreescritura de los datos del buffer de lectura se utilizan dos variables de tipo block.
    El block1 se activa desde proceso productor y avisa al consumidor que el envio de la señal ha comenzado y que por lo tanto 
    puede iniciar la adquisición. 
    El block2 se activa desde el proceso consumidor y aviso al productor que la lesctura de los datos ha finalizado y por lo tanto
    puede comenzar un nuevo paso del barrido. 
    Teniendo en cuenta que existe un retardo entre la señal enviada y adquirida, y que existe variabilidad en el retardo; se puede
    utilizar el canal 0 de entrada y salida para el envio y adquisicón de una señal de disparo que permita sincronizar las mediciones.
    Notar que cuando se pone output_channel = 1, en la segunda salida pone la misma señal que en el channel 1 de salida.
    
    Parámetros:
    -----------
    
    fs : int, frecuencia de sampleo de la placa de audio. Valor máximo 44100*8 Hz. [Hz]
    input_channels : int, cantidad de canales de entrada.
    data_out : numpy array dtype=np.float32, array de tres dimensiones de la señal a enviar [cantidad_de_pasos][muestras_por_paso][output_channels]
    corrige_retardos = {'si','no'}, corrige el retardo utilizando la función sincroniza_con_trigger
    offset_correlacion: int, muestra (tiempo) del trigger a partir de cual se hace la correlacion
    steps_correlacion: int, muestras (tiempo) del trigger con el cual se hace la correlacion
    
    Salida (returns):
    -----------------
    data_in: numpy array, array de tamaño [cantidad_de_pasos][muestras_por_pasos_input][input_channels]
    retardos: numpy_array, array de tamaño [cantidad_de_pasos] con los retardos entre trigger enviado y adquirido
    
    Las muestras_por_pasos está determinada por los tiempos de duración de la señal enviada y adquirida. El tiempo entre 
    muestras es 1/fs.
    
    Ejemplo:
    --------
    
    fs = 44100
    input_channels = 2
    data_out = np.array([[][]])   
    corrige_retardos = 'si'
    
    data_in, retardos = play_rec(parametros)
   
    Autores: Leslie Cusato, Marco Petriella    
    """  
    
    # Tipo de dato de entrada
    #dato = 'int16'
    if dato is 'int32':
        dato_np = np.int32
        dato_pyaudio = pyaudio.paInt32
    elif dato is 'int16':
        dato_np = np.int16
        dato_pyaudio = pyaudio.paInt16        
    
    # Numero de muestras originales de data_out
    original_size1 = data_out.shape[1]
    
    # Agrega Delay en data_out. Es para asegurar que el delay entre envio-adquisicion no corte la señal. Agrega ceros adelante.
    #delay = 0.0
    sample_delay = int(fs*delay)
    new_size1 = original_size1 + sample_delay
    data_out = completa_con_ceros(data_out,new_size1,mode='backward')
    
    # Parametro para obligar que el tamaño del chunk enviado sea multiplo de ind (agrega ceros al final)
    ind = 1024 
    new_size1 = int(np.ceil(new_size1/ind)*ind)
    data_out = completa_con_ceros(data_out,new_size1)
    
    #Pasos del barrido
    steps = data_out.shape[0]
    
    # Cargo parametros comunes a los dos canales  
    duration_sec_send = data_out.shape[1]/fs
    output_channels = data_out.shape[2]
            
    # Obligo a la duracion de la adquisicion > a la de salida    
    duration_sec_acq = duration_sec_send + 1 
    
    # Inicia pyaudio
    p = pyaudio.PyAudio()
    
    # Defino los buffers de lectura y escritura
    chunk_send = data_out.shape[1]
    chunk_acq = int(fs*duration_sec_acq)
          
    # Defino el stream del parlante
    stream_output = p.open(format=pyaudio.paFloat32,
                    channels = output_channels,
                    rate = fs,
                    output = True,                  
    )
    
    # Defino un buffer de lectura efectivo que tiene en cuenta el delay de la medición
    chunk_delay = int(fs*stream_output.get_output_latency()) 
    chunk_acq_eff = chunk_acq + chunk_delay
    
    # Obligo a que el tamaño del chunk adquirido sea multiplo de ind o potencia de 2
    #chunk_acq_eff = int(np.ceil(chunk_acq_eff/ind)*ind)
    chunk_acq_eff = int(2**(np.ceil(np.log(chunk_acq_eff)/np.log(2))))

    # Donde se guardan los resultados                     
    data_in = np.zeros([data_out.shape[0],chunk_acq_eff,input_channels],dtype=dato_np)    
    
    # Defino el stream del microfono
    stream_input = p.open(format = dato_pyaudio,
                    channels = input_channels,
                    rate = fs,
                    input = True,
                    frames_per_buffer = chunk_acq_eff*p.get_sample_size(dato_pyaudio),
    )
    
    # Defino los semaforos para sincronizar la señal y la adquisicion
    lock1 = threading.Lock() # Este lock es para asegurar que la adquisicion este siempre dentro de la señal enviada
    lock2 = threading.Lock() # Este lock es para asegurar que no se envie una nueva señal antes de haber adquirido y guardado la anterior
    lock1.acquire() # Inicializa el lock, lo pone en cero.
       
    # Defino el thread que envia la señal          
    def producer(steps):  
        i = 0
        while i < steps and producer_exit[0] is False:
                                      
            # Genero las señales de salida para los canales
            samples = np.zeros([output_channels,4*chunk_send],dtype = np.float32)
            for j in range(output_channels):
                    samples[j,0:chunk_send] = data_out[i,:,j]
              
            # Paso la salida a un array de una dimension
            samples_out = np.reshape(samples,4*chunk_send*output_channels,order='F')
                    
            # Se entera que se guardó el paso anterior (lock2), avisa que comienza el nuevo (lock1), y envia la señal
            lock2.acquire() 
            lock1.release() 
            stream_output.start_stream()
            stream_output.write(samples_out)
            stream_output.stop_stream()     
            
            i = i + 1
    
        producer_exit[0] = True  
            

    # Defino el thread que adquiere la señal   
    def consumer(steps):
        tiempo_ini = datetime.datetime.now()
        i = 0
        while i < steps and consumer_exit[0] is False:
            
            # Toma el lock, adquiere la señal y la guarda en el array
            lock1.acquire()
            stream_input.start_stream()
            data_i = stream_input.read(chunk_acq_eff)  
            stream_input.stop_stream()   
                
            data_i = -np.frombuffer(data_i, dtype=dato_np)                            
                
            # Guarda la salida                   
            for j in range(input_channels):
                data_in[i,:,j] = data_i[j::input_channels]                   
                    
            # Barra de progreso
            barra_progreso(i,steps,'Progreso barrido',tiempo_ini)   
            
            i = i + 1
                               
            lock2.release() # Avisa al productor que terminó de escribir los datos y puede comenzar con el próximo step
    
        consumer_exit[0] = True  
           
    # Variables de salida de los threads
    producer_exit = [False]   
    consumer_exit = [False] 
            
    # Inicio los threads    
    print (u'\n Inicio barrido \n Presione Ctrl + c para interrumpir.')
    t1 = threading.Thread(target=producer, args=[steps])
    t2 = threading.Thread(target=consumer, args=[steps])
    t1.start()
    t2.start()
    
    # Salida de la medición       
    salida_forzada = 0
    while not producer_exit[0] or not consumer_exit[0]:
        try: 
            time.sleep(0.2)
        except KeyboardInterrupt:
            salida_forzada = 1
            consumer_exit[0] = True  
            producer_exit[0] = True  
            time.sleep(0.2)
            print ('\n \n Medición interrumpida \n')


    # Finalizo los puertos 
    while stream_input.is_active() and stream_output.is_active():
        time.sleep(0.2)
    
    stream_input.close()
    stream_output.close()
    p.terminate()   
        
    # Corrección de retardo por correlación cruzada
    retardos = np.array([])
    if corrige_retardos is 'si' and salida_forzada == 0:            
        data_in, retardos = sincroniza_con_trigger(data_out[:,sample_delay:sample_delay+original_size1,:], data_in,offset_correlacion, steps_correlacion)       
        retardos = retardos-sample_delay
    
    return data_in, retardos
 


def play_rec1(fs_in,fs_out,input_channels,data_out,corrige_retardos,offset_correlacion=0,steps_correlacion=0,delay=0.0,dato='int32'):
    
    
    """
    Descripción:
    ------------
    Esta función permite utilizar la placa de audio de la pc como un generador de funciones / osciloscopio
    con dos canales de entrada y dos de salida. Para ello utiliza la libreria pyaudio y las opciones de write() y read()
    para escribir y leer los buffer de escritura y lectura. Para realizar el envio y adquisición simultánea de señales, utiliza
    un esquema de tipo productor-consumidor que se ejecutan en thread o hilos diferenntes. Para realizar la comunicación 
    entre threads y evitar overrun o sobreescritura de los datos del buffer de lectura se utilizan dos variables de tipo block.
    El block1 se activa desde proceso productor y avisa al consumidor que el envio de la señal ha comenzado y que por lo tanto 
    puede iniciar la adquisición. 
    El block2 se activa desde el proceso consumidor y aviso al productor que la lesctura de los datos ha finalizado y por lo tanto
    puede comenzar un nuevo paso del barrido. 
    Teniendo en cuenta que existe un retardo entre la señal enviada y adquirida, y que existe variabilidad en el retardo; se puede
    utilizar el canal 0 de entrada y salida para el envio y adquisicón de una señal de disparo que permita sincronizar las mediciones.
    Notar que cuando se pone output_channel = 1, en la segunda salida pone la misma señal que en el channel 1 de salida.
    
    Parámetros:
    -----------
    
    fs : int, frecuencia de sampleo de la placa de audio. Valor máximo 44100*8 Hz. [Hz]
    input_channels : int, cantidad de canales de entrada.
    data_out : numpy array dtype=np.float32, array de tres dimensiones de la señal a enviar [cantidad_de_pasos][muestras_por_paso][output_channels]
    corrige_retardos = {'si','no'}, corrige el retardo utilizando la función sincroniza_con_trigger
    offset_correlacion: int, muestra (tiempo) del trigger a partir de cual se hace la correlacion
    steps_correlacion: int, muestras (tiempo) del trigger con el cual se hace la correlacion
    
    Salida (returns):
    -----------------
    data_in: numpy array, array de tamaño [cantidad_de_pasos][muestras_por_pasos_input][input_channels]
    retardos: numpy_array, array de tamaño [cantidad_de_pasos] con los retardos entre trigger enviado y adquirido
    
    Las muestras_por_pasos está determinada por los tiempos de duración de la señal enviada y adquirida. El tiempo entre 
    muestras es 1/fs.
    
    Ejemplo:
    --------
    
    fs = 44100
    input_channels = 2
    data_out = np.array([[][]])   
    corrige_retardos = 'si'
    
    data_in, retardos = play_rec(parametros)
   
    Autores: Leslie Cusato, Marco Petriella    
    """  
    
    # Tipo de dato de entrada
    #dato = 'int16'
    if dato is 'int32':
        dato_np = np.int32
        dato_pyaudio = pyaudio.paInt32
    elif dato is 'int16':
        dato_np = np.int16
        dato_pyaudio = pyaudio.paInt16        
    
    # Numero de muestras originales de data_out
    original_size1 = data_out.shape[1]
    
    # Agrega Delay en data_out. Es para asegurar que el delay entre envio-adquisicion no corte la señal. Agrega ceros adelante.
    #delay = 0.0
    sample_delay = int(fs_out*delay)
    new_size1 = original_size1 + sample_delay
    data_out = completa_con_ceros(data_out,new_size1,mode='backward')
    
    # Parametro para obligar que el tamaño del chunk enviado sea multiplo de ind (agrega ceros al final)
    ind = 1024 
    new_size1 = int(np.ceil(new_size1/ind)*ind)
    data_out = completa_con_ceros(data_out,new_size1)
    
    #Pasos del barrido
    steps = data_out.shape[0]
    
    # Cargo parametros comunes a los dos canales  
    duration_sec_send = data_out.shape[1]/fs_out
    output_channels = data_out.shape[2]
            
    # Obligo a la duracion de la adquisicion > a la de salida    
    duration_sec_acq = duration_sec_send + 1 
    
    # Inicia pyaudio
    p = pyaudio.PyAudio()
    
    # Defino los buffers de lectura y escritura
    chunk_send = data_out.shape[1]
    chunk_acq = int(fs_in*duration_sec_acq)
          
    # Defino el stream del parlante
    stream_output = p.open(format=pyaudio.paFloat32,
                    channels = output_channels,
                    rate = fs_out,
                    output = True,                  
    )
    
    # Defino un buffer de lectura efectivo que tiene en cuenta el delay de la medición
    chunk_delay = int(fs_in*stream_output.get_output_latency()) 
    chunk_acq_eff = chunk_acq + chunk_delay
    
    # Obligo a que el tamaño del chunk adquirido sea multiplo de ind o potencia de 2
    #chunk_acq_eff = int(np.ceil(chunk_acq_eff/ind)*ind)
    chunk_acq_eff = int(2**(np.ceil(np.log(chunk_acq_eff)/np.log(2))))

    # Donde se guardan los resultados                     
    data_in = np.zeros([data_out.shape[0],chunk_acq_eff,input_channels],dtype=dato_np)    
    
    # Defino el stream del microfono
    stream_input = p.open(format = dato_pyaudio,
                    channels = input_channels,
                    rate = fs_in,
                    input = True,
                    frames_per_buffer = chunk_acq_eff*p.get_sample_size(dato_pyaudio),
    )
    
    # Defino los semaforos para sincronizar la señal y la adquisicion
    lock1 = threading.Lock() # Este lock es para asegurar que la adquisicion este siempre dentro de la señal enviada
    lock2 = threading.Lock() # Este lock es para asegurar que no se envie una nueva señal antes de haber adquirido y guardado la anterior
    lock1.acquire() # Inicializa el lock, lo pone en cero.
       
    # Defino el thread que envia la señal          
    def producer(steps):  
        i = 0
        while i < steps and producer_exit[0] is False:
                                      
            # Genero las señales de salida para los canales
            samples = np.zeros([output_channels,4*chunk_send],dtype = np.float32)
            for j in range(output_channels):
                    samples[j,0:chunk_send] = data_out[i,:,j]
              
            # Paso la salida a un array de una dimension
            samples_out = np.reshape(samples,4*chunk_send*output_channels,order='F')
                    
            # Se entera que se guardó el paso anterior (lock2), avisa que comienza el nuevo (lock1), y envia la señal
            lock2.acquire() 
            lock1.release() 
            stream_output.start_stream()
            stream_output.write(samples_out)
            stream_output.stop_stream()     
            
            i = i + 1
    
        producer_exit[0] = True  
            

    # Defino el thread que adquiere la señal   
    def consumer(steps):
        tiempo_ini = datetime.datetime.now()
        i = 0
        while i < steps and consumer_exit[0] is False:
            
            # Toma el lock, adquiere la señal y la guarda en el array
            lock1.acquire()
            stream_input.start_stream()
            data_i = stream_input.read(chunk_acq_eff)  
            stream_input.stop_stream()   
                
            data_i = -np.frombuffer(data_i, dtype=dato_np)                            
                
            # Guarda la salida                   
            for j in range(input_channels):
                data_in[i,:,j] = data_i[j::input_channels]                   
                    
            # Barra de progreso
            barra_progreso(i,steps,'Progreso barrido',tiempo_ini)   
            
            i = i + 1
                               
            lock2.release() # Avisa al productor que terminó de escribir los datos y puede comenzar con el próximo step
    
        consumer_exit[0] = True  
           
    # Variables de salida de los threads
    producer_exit = [False]   
    consumer_exit = [False] 
            
    # Inicio los threads    
    print (u'\n Inicio barrido \n Presione Ctrl + c para interrumpir.')
    t1 = threading.Thread(target=producer, args=[steps])
    t2 = threading.Thread(target=consumer, args=[steps])
    t1.start()
    t2.start()
    
    # Salida de la medición       
    salida_forzada = 0
    while not producer_exit[0] or not consumer_exit[0]:
        try: 
            time.sleep(0.2)
        except KeyboardInterrupt:
            salida_forzada = 1
            consumer_exit[0] = True  
            producer_exit[0] = True  
            time.sleep(0.2)
            print ('\n \n Medición interrumpida \n')


    # Finalizo los puertos 
    while stream_input.is_active() and stream_output.is_active():
        time.sleep(0.2)
    
    stream_input.close()
    stream_output.close()
    p.terminate()   
        
    # Corrección de retardo por correlación cruzada
    retardos = np.array([])
    if corrige_retardos is 'si' and salida_forzada == 0:            
        data_in, retardos = sincroniza_con_trigger(data_out[:,sample_delay:sample_delay+original_size1,:], data_in,offset_correlacion, steps_correlacion)       
        retardos = retardos-sample_delay
    
    return data_in, retardos



def play_rec_nidaqmx(fs_in,fs_out,input_channels,data_out,corrige_retardos,offset_correlacion=0,steps_correlacion=0,delay=0.0,dato='int32'):
    
    
    """
    Descripción:
    ------------
    Esta función permite utilizar la placa de audio de la pc como un generador de funciones / osciloscopio
    con dos canales de entrada y dos de salida. Para ello utiliza la libreria pyaudio y las opciones de write() y read()
    para escribir y leer los buffer de escritura y lectura. Para realizar el envio y adquisición simultánea de señales, utiliza
    un esquema de tipo productor-consumidor que se ejecutan en thread o hilos diferenntes. Para realizar la comunicación 
    entre threads y evitar overrun o sobreescritura de los datos del buffer de lectura se utilizan dos variables de tipo block.
    El block1 se activa desde proceso productor y avisa al consumidor que el envio de la señal ha comenzado y que por lo tanto 
    puede iniciar la adquisición. 
    El block2 se activa desde el proceso consumidor y aviso al productor que la lesctura de los datos ha finalizado y por lo tanto
    puede comenzar un nuevo paso del barrido. 
    Teniendo en cuenta que existe un retardo entre la señal enviada y adquirida, y que existe variabilidad en el retardo; se puede
    utilizar el canal 0 de entrada y salida para el envio y adquisicón de una señal de disparo que permita sincronizar las mediciones.
    Notar que cuando se pone output_channel = 1, en la segunda salida pone la misma señal que en el channel 1 de salida.
    
    Parámetros:
    -----------
    
    fs : int, frecuencia de sampleo de la placa de audio. Valor máximo 44100*8 Hz. [Hz]
    input_channels : int, cantidad de canales de entrada.
    data_out : numpy array dtype=np.float32, array de tres dimensiones de la señal a enviar [cantidad_de_pasos][muestras_por_paso][output_channels]
    corrige_retardos = {'si','no'}, corrige el retardo utilizando la función sincroniza_con_trigger
    offset_correlacion: int, muestra (tiempo) del trigger a partir de cual se hace la correlacion
    steps_correlacion: int, muestras (tiempo) del trigger con el cual se hace la correlacion
    
    Salida (returns):
    -----------------
    data_in: numpy array, array de tamaño [cantidad_de_pasos][muestras_por_pasos_input][input_channels]
    retardos: numpy_array, array de tamaño [cantidad_de_pasos] con los retardos entre trigger enviado y adquirido
    
    Las muestras_por_pasos está determinada por los tiempos de duración de la señal enviada y adquirida. El tiempo entre 
    muestras es 1/fs.
    
    Ejemplo:
    --------
    
    fs = 44100
    input_channels = 2
    data_out = np.array([[][]])   
    corrige_retardos = 'si'
    
    data_in, retardos = play_rec(parametros)
   
    Autores: Leslie Cusato, Marco Petriella    
    """  
    
    # Tipo de dato de entrada
    #dato = 'int16'
    if dato is 'int32':
        dato_np = np.int32
        dato_pyaudio = pyaudio.paInt32
    elif dato is 'int16':
        dato_np = np.int16
        dato_pyaudio = pyaudio.paInt16        
    
    # Numero de muestras originales de data_out
    original_size1 = data_out.shape[1]
    
    # Agrega Delay en data_out. Es para asegurar que el delay entre envio-adquisicion no corte la señal. Agrega ceros adelante.
    #delay = 0.0
    sample_delay = int(fs_out*delay)
    new_size1 = original_size1 + sample_delay
    data_out = completa_con_ceros(data_out,new_size1,mode='backward')
    
    # Parametro para obligar que el tamaño del chunk enviado sea multiplo de ind (agrega ceros al final)
    ind = 1024 
    new_size1 = int(np.ceil(new_size1/ind)*ind)
    data_out = completa_con_ceros(data_out,new_size1)
    
    #Pasos del barrido
    steps = data_out.shape[0]
    
    # Cargo parametros comunes a los dos canales  
    duration_sec_send = data_out.shape[1]/fs_out
    output_channels = data_out.shape[2]
            
    # Obligo a la duracion de la adquisicion > a la de salida    
    duration_sec_acq = duration_sec_send + 1 
    
    # Inicia pyaudio
    p = pyaudio.PyAudio()
    
    # Defino los buffers de lectura y escritura
    chunk_send = data_out.shape[1]
    chunk_acq = int(fs_in*duration_sec_acq)
          
    # Defino el stream del parlante
    stream_output = p.open(format=pyaudio.paFloat32,
                    channels = output_channels,
                    rate = fs_out,
                    output = True,                  
    )
    
    # Defino un buffer de lectura efectivo que tiene en cuenta el delay de la medición
    chunk_delay = int(fs_in*stream_output.get_output_latency()) 
    chunk_acq_eff = chunk_acq + chunk_delay
    
    # Obligo a que el tamaño del chunk adquirido sea multiplo de ind o potencia de 2
    #chunk_acq_eff = int(np.ceil(chunk_acq_eff/ind)*ind)
    chunk_acq_eff = int(2**(np.ceil(np.log(chunk_acq_eff)/np.log(2))))

    # Donde se guardan los resultados                     
    data_in = np.zeros([data_out.shape[0],chunk_acq_eff,input_channels],dtype=np.float64)    
    
    # Defino el stream del microfono
#    stream_input = p.open(format = dato_pyaudio,
#                    channels = input_channels,
#                    rate = fs,
#                    input = True,
#                    frames_per_buffer = chunk_acq_eff*p.get_sample_size(dato_pyaudio),
#    )
    
    # Defino los semaforos para sincronizar la señal y la adquisicion
    lock1 = threading.Lock() # Este lock es para asegurar que la adquisicion este siempre dentro de la señal enviada
    lock2 = threading.Lock() # Este lock es para asegurar que no se envie una nueva señal antes de haber adquirido y guardado la anterior
    lock1.acquire() # Inicializa el lock, lo pone en cero.
       
    # Defino el thread que envia la señal          
    def producer(steps):  
        i = 0
        while i < steps and producer_exit[0] is False:
                                      
            # Genero las señales de salida para los canales
            samples = np.zeros([output_channels,4*chunk_send],dtype = np.float32)
            for j in range(output_channels):
                    samples[j,0:chunk_send] = data_out[i,:,j]
              
            # Paso la salida a un array de una dimension
            samples_out = np.reshape(samples,4*chunk_send*output_channels,order='F')
                    
            # Se entera que se guardó el paso anterior (lock2), avisa que comienza el nuevo (lock1), y envia la señal
            lock2.acquire() 
            lock1.release() 
            stream_output.start_stream()
            stream_output.write(samples_out)
            stream_output.stop_stream()     
            
            i = i + 1
    
        producer_exit[0] = True  
            

    # Defino el thread que adquiere la señal   
    def consumer(steps):
        tiempo_ini = datetime.datetime.now()
        i = 0
        while i < steps and consumer_exit[0] is False:
            
            # Toma el lock, adquiere la señal y la guarda en el array
            lock1.acquire()
            
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan("Dev1/ai4",max_val=10., min_val=-10.,name_to_assign_to_channel="ch1",terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
                task.timing.cfg_samp_clk_timing(fs_in,samps_per_chan=chunk_acq_eff)
                data_i=task.read(number_of_samples_per_channel=chunk_acq_eff)                             
            
            # Guarda la salida                   
            for j in range(input_channels):
                data_in[i,:,j] = data_i[j::input_channels]                   
                    
            # Barra de progreso
            barra_progreso(i,steps,'Progreso barrido',tiempo_ini)   
            
            i = i + 1
                               
            lock2.release() # Avisa al productor que terminó de escribir los datos y puede comenzar con el próximo step
    
        consumer_exit[0] = True  
           
    # Variables de salida de los threads
    producer_exit = [False]   
    consumer_exit = [False] 
            
    # Inicio los threads    
    print (u'\n Inicio barrido \n Presione Ctrl + c para interrumpir.')
    t1 = threading.Thread(target=producer, args=[steps])
    t2 = threading.Thread(target=consumer, args=[steps])
    t1.start()
    t2.start()
    
    # Salida de la medición       
    salida_forzada = 0
    while not producer_exit[0] or not consumer_exit[0]:
        try: 
            time.sleep(0.2)
        except KeyboardInterrupt:
            salida_forzada = 1
            consumer_exit[0] = True  
            producer_exit[0] = True  
            time.sleep(0.2)
            print ('\n \n Medición interrumpida \n')


    # Finalizo los puertos 
    while stream_output.is_active():
        time.sleep(0.2)
    
    stream_output.close()
    p.terminate()   
        
    # Corrección de retardo por correlación cruzada
    retardos = np.array([])
    if corrige_retardos is 'si' and salida_forzada == 0:            
        data_in, retardos = sincroniza_con_trigger(data_out[:,sample_delay:sample_delay+original_size1,:], data_in,offset_correlacion, steps_correlacion)       
        retardos = retardos-sample_delay
    
    return data_in, retardos






def callback_pid(i, input_buffer, output_buffer_duty_cycle, output_buffer_pid_terminos, output_buffer_mean_data, output_buffer_error_data, output_buffer_pid_constants, buffer_chunks, callback_pid_variables):
    
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
        # output_buffer_error_data: np.array [buffer_chunks]. La i-posicion es la actual: output_buffer_mean_data[i,0] - lsetpoint[0].
        # buffer_chunks: Cantidad de chunks del buffer
        # output_buffer_pid_constants [buffer_chunks, 5] - > [buffer_chunks, setpoint kp ki kd isteps]
        #   setpoint: Valor de tensión del setpoint
        #   kp: constante multiplicativa del PID
        #   ki: constante integral del PID
        #   kd: constante derivativa del PID
        #   isteps: cantidad de pasos para atrás utilizados para el termino integral
        #
        # Variables de entrada del usuario:
        # --------------------------------
        # callback_pid_variables: lista con variables puestas por el usuario
        #   Ejemplo:
        #   variable0 = callback_pid_variables[0]
        #   variable1 = callback_pid_variables[1]
        #
        # Salidas:
        # -------
        # output_buffer_duty_cycle_i: duty cycle calculado en el callback
        # termino_p: termino multiplicativo que acompaña a kp
        # termino_i: termino integral que acompaña a ki
        # termino_d: termino derivatico que acompaña a kd
        #####################################
        """
        
        # Valores maximos y minimos de duty cycle
        max_duty_cycle = 0.999
        min_duty_cycle = 0.001  
        
        setpoint = output_buffer_pid_constants[i,0]
        kp = output_buffer_pid_constants[i,1]
        ki = output_buffer_pid_constants[i,2]
        kd = output_buffer_pid_constants[i,3]
        isteps = int(output_buffer_pid_constants[i,4])
    
        # Paso anterior de buffer circular
        j = (i-1)%buffer_chunks
                
        # n-esimo paso anterior de buffer circular
        k = (i-isteps)%buffer_chunks    
        
        # Algoritmo PID
        termino_p = output_buffer_error_data[i]
        termino_d = output_buffer_error_data[i]-output_buffer_error_data[j]
        
        # termino integral
        termino_i = output_buffer_pid_terminos[j,1]
        isteps_ant = output_buffer_pid_constants[j,4]
        if isteps != isteps_ant:
            if k >= i:
                termino_i = np.sum(output_buffer_error_data[k:buffer_chunks]) + np.sum(output_buffer_error_data[0:i])
            else:
                termino_i = np.sum(output_buffer_error_data[k:i])
            termino_i = termino_i/isteps
        else:
            termino_i += output_buffer_error_data[i]/isteps - output_buffer_error_data[k]/isteps
        
        output_buffer_duty_cycle_i = output_buffer_duty_cycle[j] + kp*termino_p + ki*termino_i + kd*termino_d
        #time.sleep(0.015)
    
        # Salida de la función
        output_buffer_duty_cycle_i = min(output_buffer_duty_cycle_i,max_duty_cycle)
        output_buffer_duty_cycle_i = max(output_buffer_duty_cycle_i,min_duty_cycle)
                
        return output_buffer_duty_cycle_i, termino_p, termino_i, termino_d







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
            path_data_save + '_pid_constants.bin' : archivo binario con las constantes PID. Es un un array de dos dimensiones [:,kp ki kp isteps]
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
    
    ##### Callbacks de error ########
    def exit_callback(event):      
        evento_salida.set()
        acquiring_error.append('Medición interrumpida por el usuario')

    def exit_callback1(error_string):        
        evento_salida.set()
        acquiring_error.append(error_string)

    def warning_callback(warning_string):        
        evento_warning.set()
        warnings.append(warning_string)    
                
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
    ax.set_ylim([0,5])
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

    xi = 1.63
    ax2.text(xi,yi,'PID parameters',fontsize=default_fontsize,va='center',transform = ax2.transAxes)
    
    xi_s = 0.69
    yi_s = 0.185
    dyi_s = 0.03

    # Mensje de error
    x_error = 1.64
    y_error = 0.08

    texto_error = ax2.text(x_error,y_error,'',fontsize=default_fontsize-1,va='center',transform = ax2.transAxes,color='red') 
    
    def print_error(s):       
        s_tot = ''
        for i in range(len(s)):
            s_tot = s_tot + s[i] + '\n' 
        texto_error.set_text(s_tot)    
    
    ####### FIN PLOT ############
    #############################    


    # Defino los buffers de entrada y salida
    input_buffer = np.zeros([buffer_chunks,ai_samples,ai_nbr_channels])
    output_buffer_mean_data = np.zeros([buffer_chunks,ai_nbr_channels])
    output_buffer_duty_cycle = np.ones(buffer_chunks)*initial_do_duty_cycle
    output_buffer_error_data = np.zeros(buffer_chunks)
    output_buffer_pid_constants = np.zeros([buffer_chunks,5])    
    output_buffer_pid_terminos = np.zeros([buffer_chunks,3])    
       
    # Semaforos
    semaphore1 = threading.Semaphore(0) # Input buffer
    semaphore2 = threading.Semaphore(0) # Output buffer
    semaphore3 = threading.Semaphore(0) # Guardado de raw data
    semaphore4 = threading.Semaphore(0) # Guardado de processed data
    semaphore5 = threading.Semaphore(0) # Plot
    
    # Inicializo variables de interfaz: setpoint, kp, ki, kd, isteps, pid_on_off
    lsetpoint = [setpoint]
    lkp = [kp]
    lki = [ki]
    lkd = [kd]
    listeps = [isteps]
    pid_onoff_button = [True]
    ##############################################################################
    
    
    
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


                
    # Thread del callback        
    def callback_thread():
        
        global lsetpoint, lkp, lki, lkd, listeps, pid_onoff_button
        lsetpoint = [setpoint]
        lkp = [kp]
        lki = [ki]
        lkd = [kd]
        listeps = [isteps]
        pid_onoff_button = [True]        
        
              
        i = 0
        while not evento_salida.is_set(): 

            if semaphore1._value > buffer_chunks:
                error_string = 'Hay overrun en llenado del input_buffer!'
                exit_callback1(error_string)        
                            
            if semaphore2._value > buffer_chunks:
                error_string = 'Hay overrun en el vaciado del output_buffer!'
                exit_callback1(error_string)            
            
            semaphore1.acquire()    
    
            ## Inicio Callback      
            output_buffer_mean_data[i,:] = np.mean(input_buffer[i,:,:],axis=0)
            output_buffer_error_data[i] = output_buffer_mean_data[i,0] - lsetpoint[0]  
            output_buffer_pid_constants[i,0] = lsetpoint[0] 
            output_buffer_pid_constants[i,1] = lkp[0] 
            output_buffer_pid_constants[i,2] = lki[0] 
            output_buffer_pid_constants[i,3] = lkd[0] 
            output_buffer_pid_constants[i,4] = listeps[0]
            output_buffer_duty_cycle_i, termino_p, termino_i, termino_d = callback_pid(i, input_buffer, output_buffer_duty_cycle, output_buffer_pid_terminos, output_buffer_mean_data, output_buffer_error_data, output_buffer_pid_constants, buffer_chunks, callback_pid_variables)             
            if pid_onoff_button[0] is False:
                output_buffer_duty_cycle_i = initial_do_duty_cycle
            output_buffer_duty_cycle[i] = output_buffer_duty_cycle_i       
            output_buffer_pid_terminos[i,:] = np.array([termino_p, termino_i, termino_d])
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
        global ssetpoint, skp, ski, skd,sisteps, lsetpoint, lkp, lki, lkd, listeps, bonoff
                    
        
        lsetpoint = [setpoint]
         
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
        axsetpoint = plt.axes([xi_s, yi_s, 0.10, 0.02], facecolor=axcolor)
        ssetpoint = Slider(axsetpoint, 'Setpoint',0.001, 5.0, valinit=setpoint)
        ssetpoint.on_changed(update_pid) 
        ssetpoint.label.set_size(default_fontsize)
        ssetpoint.valtext.set_size(default_fontsize)
    
        # Slider kp
        axkp = plt.axes([xi_s, yi_s - 1*dyi_s, 0.10, 0.02], facecolor=axcolor)
        skp = Slider(axkp, 'kp',0.0, 2.0, valinit=kp)
        skp.on_changed(update_pid) 
        skp.label.set_size(default_fontsize)
        skp.valtext.set_size(default_fontsize)
    
        # Slider ki
        axki = plt.axes([xi_s, yi_s - 2*dyi_s, 0.10, 0.02], facecolor=axcolor)
        ski = Slider(axki, 'ki',0.0, 2.0, valinit=ki)
        ski.on_changed(update_pid) 
        ski.label.set_size(default_fontsize)
        ski.valtext.set_size(default_fontsize)
    
        # Slider integral steps
        axisteps = plt.axes([xi_s + 0.17, yi_s - 2*dyi_s, 0.10, 0.02], facecolor=axcolor)
        sisteps = Slider(axisteps, 'steps',0.0, buffer_chunks, valinit=isteps)
        sisteps.valfmt = '%2d'
        sisteps.on_changed(update_pid) 
        sisteps.label.set_size(default_fontsize)
        sisteps.valtext.set_size(default_fontsize)
    
        
        # Slider kd
        axkd = plt.axes([xi_s, yi_s - 3*dyi_s, 0.10, 0.02], facecolor=axcolor)
        skd = Slider(axkd, 'kd',0.0, 2.0, valinit=kd)
        skd.on_changed(update_pid)   
        skd.label.set_size(default_fontsize)
        skd.valtext.set_size(default_fontsize)          
        ############### FIN BOTONES ######################        
        
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
    
                setpoint_line.set_ydata(lsetpoint[0])
                
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
                             
                
                if evento_warning.is_set():                
                    print_error(warnings)
                    warnings = []
                    evento_warning.clear() 
                    
                fig.canvas.draw_idle()
                   
            i = i+1
            i = i%buffer_chunks 

        print_error(acquiring_error)
        fig.canvas.draw_idle()
 

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
        



def save_to_np_file(filename,arr):
    f_handle = open(filename, 'ab')
    np.save(f_handle, arr)
    f_handle.close()    


def load_from_np_file(filename):

    f = open(filename, 'rb')
    arr = np.load(f)  
    while True:
        try:
            arr = np.append(arr,np.load(f),axis=0)
        except:
            break
    f.close()  

    return arr    



def completa_con_ceros(data_out,new_size1,mode='forward'):
    
    
    data_out_corrected = np.zeros([data_out.shape[0],new_size1,data_out.shape[2]],dtype=data_out.dtype)
    
    if mode is 'forward':
        for i in range(data_out.shape[0]):
            for k in range(data_out.shape[2]):                
                data_out_corrected[i,0:data_out.shape[1],k] = data_out[i,:,k]
                
    elif mode is 'backward':
        for i in range(data_out.shape[0]):
            for k in range(data_out.shape[2]):               
                data_out_corrected[i,data_out_corrected.shape[1]-data_out.shape[1]:,k] = data_out[i,:,k]                
    
    return data_out_corrected


def sincroniza_con_trigger(trigger,data_in,offset_correlacion=0,steps_correlacion=0, ch=0):
    
    """
    Esta función corrige el retardo de las mediciones adquiridas con la función play_rec. Para ello utiliza la señal de 
    trigger enviada y adquirida en el canal 0 de la placa de audio, y sincroniza las mediciones de todos los canales de entrada. 
    El retardo se determina a partir de realizar la correlación cruzada entre la señal enviada y adquirida, y encontrando la posición
    del máximo del resultado.
    
    
    Parámetros:
    -----------
    trigger: numpy array, array de tamaño [cantidad_de_pasos][muestras_por_pasos_trigger][trigger_channels]
    data_in: numpy array, array de tamaño [cantidad_de_pasos][muestras_por_pasos_input][input_channels]
    offset_correlacion: int, muestra (tiempo) del trigger a partir de cual se hace la correlacion
    steps_correlacion: int, muestras (tiempo) del trigger con el cual se hace la correlacion
    
    Salida (returns):
    -----------------
    data_in_corrected : numpy array, señal de salida con retardo corregido de tamaño [cantidad_de_pasos][muestras_por_pasos_trigger][input_channels]. 
                         El tamaño de la segunda dimensión es la misma que la de data_trigger.
    retardos : numpy array, array con los retardos de tamaño [cantidad_de_pasos].
    
    Autores: Leslie Cusato, Marco Petriella   
    """
    
    print (u'\n Inicio corrección \n Presione Ctrl + c para interrumpir.')
 
#    trigger = trigger.astype(np.float32)
#    data_in = data_in.astype(np.float32)
    
    # Cantidad de muestras extras que se toman
    extra = 0
    
    # Estas son las salidas    
    data_in_corrected = np.zeros([trigger.shape[0],trigger.shape[1]+extra,data_in.shape[2]])
    retardos = np.array([])
    
    # Defino la matriz de trigger enviada y adquirida
    trigger_send = trigger[:,:,ch]
    trigger_acq = data_in[:,:,ch]  

    # Array donde se guarda la señal de trigger digital
    comp = np.zeros(trigger_acq.shape[1])  

    if steps_correlacion == 0:
        steps_correlacion = trigger_send.shape[1]     
    
    tiempo_ini = datetime.datetime.now()
    errores = []         
    for i in range(data_in.shape[0]):
        try:
            
            # Correlacion con la función de numpy
#            corr = np.correlate(trigger_acq[i,:] - np.mean(trigger_acq[i,:]),trigger_send[i,offset_correlacion:offset_correlacion+steps_correlacion] - np.mean(trigger_send[i,offset_correlacion:offset_correlacion+steps_correlacion]))
#            pos_max = np.argmax(corr) - offset_correlacion
            
            # Uso correlación cruzada con FFT que es mucho mas rapida que la de numpy
            comp[0:steps_correlacion] = trigger_send[i,offset_correlacion:offset_correlacion+steps_correlacion]
            corr = cross_correlation_using_fft(trigger_acq[i,:] - np.mean(trigger_acq[i,:]),comp-np.mean(comp))
            pos_max = np.argmax(corr)-int(len(corr)/2)-offset_correlacion+1
            
            retardos = np.append(retardos,pos_max)
#            plt.plot(corr)
#            print(pos_max)

            
            barra_progreso(i,data_in.shape[0],u'Progreso corrección',tiempo_ini) 
            
            if pos_max >= 0 and pos_max+trigger_send.shape[1]+extra < data_in.shape[1]:             
                for j in range(data_in.shape[2]):
                    data_in_corrected[i,:,j] = data_in[i,pos_max:pos_max+trigger_send.shape[1]+extra,j]
            else:
                errores.append(i)
                for j in range(data_in.shape[2]):
                    data_in_corrected[i,:,j] = np.full_like(data_in_corrected[i,:,j], np.nan)
                    
        except KeyboardInterrupt:
            print (u'\n \n Proceso corrección interrumpido \n')
            break

                
    for i in errores:
        print(u'- Correlación fuera de los límites en el paso ' + str(i) + '. Atención! la salida se completa con NaNs. \n')
        
        
    return data_in_corrected, retardos



def sincroniza_con_trigger1(trigger,data_in,offset_correlacion=0,steps_correlacion=0, ch=0):
    
    """
    Esta funcion es solo para poder graficar la correlacion es la misma que la de arriba.
    Esta función corrige el retardo de las mediciones adquiridas con la función play_rec. Para ello utiliza la señal de 
    trigger enviada y adquirida en el canal 0 de la placa de audio, y sincroniza las mediciones de todos los canales de entrada. 
    El retardo se determina a partir de realizar la correlación cruzada entre la señal enviada y adquirida, y encontrando la posición
    del máximo del resultado.
    
    
    Parámetros:
    -----------
    trigger: numpy array, array de tamaño [cantidad_de_pasos][muestras_por_pasos_trigger][trigger_channels]
    data_in: numpy array, array de tamaño [cantidad_de_pasos][muestras_por_pasos_input][input_channels]
    offset_correlacion: int, muestra (tiempo) del trigger a partir de cual se hace la correlacion
    steps_correlacion: int, muestras (tiempo) del trigger con el cual se hace la correlacion
    
    Salida (returns):
    -----------------
    data_in_corrected : numpy array, señal de salida con retardo corregido de tamaño [cantidad_de_pasos][muestras_por_pasos_trigger][input_channels]. 
                         El tamaño de la segunda dimensión es la misma que la de data_trigger.
    retardos : numpy array, array con los retardos de tamaño [cantidad_de_pasos].
    
    Autores: Leslie Cusato, Marco Petriella   
    """
    
    print (u'\n Inicio corrección \n Presione Ctrl + c para interrumpir.')
 
#    trigger = trigger.astype(np.float32)
#    data_in = data_in.astype(np.float32)
    
    # Cantidad de muestras extras que se toman
    extra = 0
    
    # Estas son las salidas    
    data_in_corrected = np.zeros([trigger.shape[0],trigger.shape[1]+extra,data_in.shape[2]])
    retardos = np.array([])
    
    # Defino la matriz de trigger enviada y adquirida
    trigger_send = trigger[:,:,ch]
    trigger_acq = data_in[:,:,ch]  

    # Array donde se guarda la señal de trigger digital
    comp = np.zeros(trigger_acq.shape[1])  

    if steps_correlacion == 0:
        steps_correlacion = trigger_send.shape[1]     
    
    tiempo_ini = datetime.datetime.now()
    errores = []         
    for i in range(data_in.shape[0]):
        try:
            
            # Correlacion con la función de numpy
#            corr = np.correlate(trigger_acq[i,:] - np.mean(trigger_acq[i,:]),trigger_send[i,offset_correlacion:offset_correlacion+steps_correlacion] - np.mean(trigger_send[i,offset_correlacion:offset_correlacion+steps_correlacion]))
#            pos_max = np.argmax(corr) - offset_correlacion
            
            # Uso correlación cruzada con FFT que es mucho mas rapida que la de numpy
            comp[0:steps_correlacion] = trigger_send[i,offset_correlacion:offset_correlacion+steps_correlacion]
            corr = cross_correlation_using_fft(trigger_acq[i,:] - np.mean(trigger_acq[i,:]),comp-np.mean(comp))
            pos_max = np.argmax(corr)-int(len(corr)/2)-offset_correlacion+1
            
            retardos = np.append(retardos,pos_max)
#            plt.plot(corr)
#            print(pos_max)

            
            barra_progreso(i,data_in.shape[0],u'Progreso corrección',tiempo_ini) 
            
            if pos_max >= 0 and pos_max+trigger_send.shape[1]+extra < data_in.shape[1]:             
                for j in range(data_in.shape[2]):
                    data_in_corrected[i,:,j] = data_in[i,pos_max:pos_max+trigger_send.shape[1]+extra,j]
            else:
                errores.append(i)
                for j in range(data_in.shape[2]):
                    data_in_corrected[i,:,j] = np.full_like(data_in_corrected[i,:,j], np.nan)
                    
        except KeyboardInterrupt:
            print (u'\n \n Proceso corrección interrumpido \n')
            break

                
    for i in errores:
        print(u'- Correlación fuera de los límites en el paso ' + str(i) + '. Atención! la salida se completa con NaNs. \n')
        
        
    return data_in_corrected, retardos, corr


def par2ind(par_level,parlante_levels):
    
    for i in range(parlante_levels.shape[0]):
        
        if parlante_levels[i] == par_level:
            break
        
    return i
    

def fft_power_density(data_vec,fs):
    
    psdx = abs(fft.fft(data_vec))**2/int(data_vec.shape[0])/fs
    psdx = psdx[0:int(data_vec.shape[0]/2)]
    psdx[1:] = 2*psdx[1:]
    
    freq = np.fft.fftfreq(data_vec.shape[0], d=1/fs)
    freq = freq[0:int(data_vec.shape[0]/2)]
    
    return freq, psdx