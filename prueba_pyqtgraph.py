# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 21:42:46 2018

@author: Marco
"""


from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

win = pg.GraphicsWindow(title="Basic plotting examples")
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)







p6 = win.addPlot(title="Updating plot")
curve1 = p6.plot(pen='g')
curve2 = p6.plot(pen='y')


data = np.random.normal(size=(10,1000))
ptr = 0


def update():
    global curve1,curve2, data, ptr, p6
    curve1.setData(data[ptr%10])
    curve2.setData(data[(ptr+4)%10])
    if ptr == 0:
        p6.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
    ptr += 1
    
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)

