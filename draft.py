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
    a = np.array([1,2,3,4,5,6])
    b = np.array([1,2,3,4])
    print(np.convolve(a,b))