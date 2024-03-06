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

class Test(object):
    def __init__(self):
        filename = 'C:/Wenlong Li/TUM/Master/第一学期/FP/Data/Hao/2023-09-06_17h50_Hao_ramp_30MVC_and_sin/X_Iocz20230906173031_03_Hao_ramp_30MVC_and_sin.otb+'
        data, channel_df = data_readers.import_otb(filename)
        self.EMG_instance = EMG_Signal.prepare(data,ElectrodeMappings,channel_df)
    
    def OutlinerTest(self,EMG_Signal):
        EMG_Signal.segment(t_Epoch=3)
        test_epoch = EMG_Signal.Epoch[100]

        p = preprocessing.Outliner_detection(test_epoch,EMG_Signal.metadata,0.5)
        p.outliner_detection()  
        p.reference_selection()
        p.threshold_calculation(7.1,2.5,6)
        a = p.detect()
        p.OutlinerVisulize()
        p.visulize(p.data,p.fs,selected=[5,6,4,12,13,14])
        plt.show()        

    def DetrendTest(self,EMG_Signal,method='median',**kwargs):
        EMG_Signal.segment(t_Epoch=3)
        test_epoch = EMG_Signal.Epoch[20]

        model = preprocessing.Detrend(test_epoch,method=method,**kwargs)
        baseline,residual = model.apply()
        print(baseline[2,:].mean())
        print(residual[2,:].mean())
        print(test_epoch[2,:].mean())
        model.visulize(baseline,EMG_Signal.fs,selected=[2,3],title='baseline')
        model.visulize(residual,EMG_Signal.fs,selected=[2,3],title='residual')
        model.visulize(test_epoch,EMG_Signal.fs,selected=[2,3],title='raw')
        plt.show()
    
    def CCATest(self,EMG_Signal):
        pass







if __name__ == '__main__':
    t = Test()
    t.DetrendTest(t.EMG_instance[0],method='qvr',lamda=5)