# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 12:36:24 2018

@author: Marco
"""



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
import nidaqmx
import nidaqmx.constants as constants
import nidaqmx.stream_writers
#import scipy.fftpack as fft 
    
params = {'legend.fontsize': 'medium',
     #     'figure.figsize': (15, 5),
         'axes.labelsize': 'medium',
         'axes.titlesize':'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)


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


def pid_daqmx(parametros):

    buffer_chunks = parametros['buffer_chunks']
    
    ai_channels = parametros['ai_channels']
    ai_samples = parametros['ai_samples']
    ai_samplerate = parametros['ai_samplerate']
    
    callback = parametros['callback']
    callback_variables = parametros['callback_variables']
    pid_variables = parametros['pid_variables']
    
    initial_pid_duty_cycle = parametros['initial_pid_duty_cycle']
    initial_pid_frequency = parametros['initial_pid_frequency']
    
    # Defino los buffers
    input_buffer = np.zeros([buffer_chunks,ai_samples,ai_channels])
    output_buffer_duty_cycle = np.ones(buffer_chunks)*initial_pid_duty_cycle
    output_buffer_frequency = np.ones(buffer_chunks)*initial_pid_frequency
          
    # Semaforos
    semaphore1 = threading.Semaphore(0)
    semaphore2 = threading.Semaphore(0)
           
    # Defino el thread que envia la señal          
    def producer_thread():  
        
        with nidaqmx.Task() as task_do:
            
            task_do.co_channels.add_co_pulse_chan_freq(counter='Dev1/ctr0',duty_cycle=initial_pid_duty_cycle,freq=initial_pid_frequency,units=nidaqmx.constants.FrequencyUnits.HZ)
            task_do.timing.cfg_implicit_timing(sample_mode=constants.AcquisitionType.CONTINUOUS)    
            digi_s = nidaqmx.stream_writers.CounterWriter(task_do.out_stream)
            task_do.start()
                       
            i = 0
            while producer_exit[0] is False:
                
                semaphore2.acquire()   
    
                digi_s.write_one_sample_pulse_frequency(frequency = output_buffer_frequency[i], duty_cycle = output_buffer_duty_cycle[i])
                
                i = i+1
                i = i%buffer_chunks     
                
    
    # Defino el thread que adquiere la señal   
    def consumer_thread():
                
        with nidaqmx.Task() as task_ai:
            task_ai.ai_channels.add_ai_voltage_chan("Dev1/ai2",max_val=5., min_val=-5.,terminal_config=constants.TerminalConfiguration.RSE)#, "Voltage")#,AIVoltageUnits.Volts)#,max_val=10., min_val=-10.)
            task_ai.timing.cfg_samp_clk_timing(ai_samplerate,samps_per_chan=ai_samples,sample_mode=constants.AcquisitionType.CONTINUOUS)
                
            i = 0
            while consumer_exit[0] is False:
        
                medicion = task_ai.read(number_of_samples_per_channel=ai_samples)
                medicion = np.asarray(medicion)
                medicion = np.reshape(medicion,ai_channels*ai_samples,order='F')
                
                for j in range(ai_channels):
                    input_buffer[i,:,j] = medicion[j:ai_channels:]  
                
                semaphore1.release() 
                
                i = i+1
                i = i%buffer_chunks                  

                
    def callback_thread():

        i = 0
        while callback_exit[0] is False:        

            if semaphore1._value > buffer_chunks:
                print('Hay overun en la lectura! \n')
            
            if semaphore2._value > buffer_chunks:
                print('Hay overun en la escritura! \n')
            
            semaphore1.acquire()    
    
            ## Inicio Callback             
            output_buffer_duty_cycle_i, output_buffer_frequency_i = callback(i, input_buffer, output_buffer_duty_cycle, output_buffer_frequency, buffer_chunks, initial_pid_duty_cycle, initial_pid_frequency, callback_variables, pid_variables)   
            output_buffer_duty_cycle[i] = output_buffer_duty_cycle_i
            output_buffer_frequency[i] = output_buffer_frequency_i                     
            ## Fin callback
    
            semaphore2.release()
            
            i = i+1
            i = i%buffer_chunks   

                         
    # Variables de salida de los threads
    producer_exit = [False]   
    consumer_exit = [False] 
    callback_exit = [False] 
            
    # Inicio los threads    
    print (u'\n Inicio barrido \n Presione Ctrl + c para interrumpir.')
    t1 = threading.Thread(target=producer_thread, args=[])
    t2 = threading.Thread(target=consumer_thread, args=[])
    t3 = threading.Thread(target=callback_thread, args=[])
    
    t1.start()
    t2.start()
    t3.start()
    
    # Salida de la medición       
    while not producer_exit[0] or not consumer_exit[0] or not callback_exit[0]:
        try: 
            time.sleep(0.2)
        except KeyboardInterrupt:
            consumer_exit[0] = True  
            producer_exit[0] = True  
            callback_exit[0] = True 
            time.sleep(0.2)
            print ('\n \n Medición interrumpida \n')

    return input_buffer, output_buffer_duty_cycle, output_buffer_frequency



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