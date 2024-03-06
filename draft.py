import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
import preprocessing
from __init__ import * 
import FeatureExtraction
import data_readers
from collections.abc import Iterable
from otb_matrices import otb_patch_map
from ElectrodeMapping import ElectrodeMappings
from matplotlib.patches import Circle,Wedge


if __name__ == '__main__':
    def ElectrodePositionPlot(layout='Rect',**kwargs):
        if layout == 'Rect':
            row = kwargs['row']
            col = kwargs['col']
            try:
                radius = kwargs['r']
            except:
                radius = 0.2
            try:
                distance = kwargs['d']
            except:
                distance = (1,0.6)
            return RectLayout(row,col,radius,distance)
    
    
    def RectLayout(row,col,r,d):
        map = ElectrodeMappings['GR10MM0808']['ElChannelMap']
        fig = plt.figure()
        ax = fig.add_subplot()
        row_d = d[0]
        col_d = d[1]
        y = np.arange(-row*row_d,0,row_d)
        x = np.arange(-col*col_d,0,col_d)
        y = y[::-1]
        CircleList = {}
        for i1,yi in enumerate(y):
            for i2,xi in enumerate(x):
                W1 = Wedge((xi,yi),r,-30,90,fc='r')
                W2 = Wedge((xi,yi),r,90,210,fc='b')
                W3 = Wedge((xi,yi),r,210,-30,fc='y')
                #C = Circle((xi,yi),radius=r,fill=False)
                ax.text(xi-0.15,yi+0.3,f'{map[i1,i2]-1}')
                #ax.add_patch(C)
                #CircleList[map[i1,i2]-1] = C
                ax.add_patch(W1)
                ax.add_patch(W2)
                ax.add_patch(W3)
        plt.axis('off')
        ax.legend()
        ax.axis('equal')
        return CircleList

    def change_color(CList,list):
        for i in list:
            CList[i].set_fill(True)
            CList[i].set_color('r')



    CList = ElectrodePositionPlot(row=8,col=8)
    #change_color(CList,[0,3,4])
    plt.show()