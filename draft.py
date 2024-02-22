import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
import preprocessing
from __init__ import * 
import FeatureExtraction
import data_readers
from collections.abc import Iterable
from otb_matrices import otb_patch_map
from ElectrodeMapping import *

if __name__ == '__main__':
    filename = "C:/Wenlong Li/TUM/Master/第一学期/FP/Data/Hao/2023-09-06_17h50_Hao_ramp_30MVC_and_sin/X_Iocz20230906173031_03_Hao_ramp_30MVC_and_sin.otb+"
    data, channel_df = data_readers.import_otb(filename)
    a = EMG_Signal.prepare(data,ElectrodeMapping,channel_df)
    print(len(a))
    # print(channel_df.columns)
    # fs = channel_df['fsamp'][0]
    
    # EMG = EMG_Signal(data,fs)
    # EMG.segment(t_Epoch=10)
    # EMG.visulize(EMG.Epoch[0][0:2,:],np.arange(2))
    # plt.show()
