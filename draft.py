import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
import preprocessing
from __init__ import * 
import FeatureExtraction
import data_readers

if __name__ == '__main__':
    filename = "C:/Wenlong Li/TUM/Master/第一学期/FP/Data/Hao/2023-09-06_17h50_Hao_ramp_30MVC_and_sin/X_Iocz20230906173031_03_Hao_ramp_30MVC_and_sin.otb+"
    data, channel_df = data_readers.import_otb(filename)
    fs = channel_df['fsamp'][0]
    EMG = EMG_Signal(data,fs)
    EMG.visulize(data[:30,:],np.arange(30))
    plt.show()