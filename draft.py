import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
import preprocessing
from EMG import EMG_Signal
import FeatureExtraction
import data_readers
from collections.abc import Iterable
from otb_matrices import otb_patch_map
from ElectrodeMapping import ElectrodeMappings
from matplotlib.patches import Circle,Wedge
import glob
import pandas as pd

if __name__ == '__main__':
    k1 = 3
    k2 = 5
    
    x1 = np.random.normal(loc=0,scale=1,size=(1,100))
    x2 = np.random.normal(loc=0,scale=1,size=(1,100))
    data = np.zeros((2,100))
    data[0,:] = k1*x1
    data[1,:] = k2*x2
    
    Model = preprocessing.PCA2(data)
    Model.fit()
    a = Model.compute(n_components=1)
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(a)
    plt.show()
    
