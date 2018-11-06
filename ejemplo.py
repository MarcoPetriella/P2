# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:56:11 2018

@author: Marco
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


def funcion():
    
    
    global st

    datos_tot=100
    cant_puntos=10
    data_inicial=np.random.normal(size=(cant_puntos,datos_tot))
    
    fig,ax=plt.subplots()
    line,=ax.plot(np.zeros(cant_puntos),'.')
    ax.set_ylim([-1,1])
    
    st=time.time()
    freq=50 #cantidad de updates por segundo
    
    
    def hola(i):
        

    #def update_plot(i):
        #print('chau1')
        global st
        line.set_ydata(data_inicial[:,i%datos_tot])
        if i%freq==0:
            print(i)
            print(time.time()-st)
            #print time.time()-st, ' seg cada ', freq,' updates'
            st=time.time()
    	#ax.set_title(str(i))
        return line,
    
    ani = animation.FuncAnimation(fig, hola,interval=20, blit=True)
    return ani
#        plt.show()
    


ani = funcion()


plt.show()

print(1)