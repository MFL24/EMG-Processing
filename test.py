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
from scipy import signal
import mne

class Test(object):
    def __init__(self):
        filename = 'C:/Wenlong Li/TUM/Master/第一学期/FP/Data/Hao/2023-09-06_17h50_Hao_ramp_30MVC_and_sin/X_Iocz20230906173031_03_Hao_ramp_30MVC_and_sin.otb+'
        data, channel_df = data_readers.import_otb(filename)
        self.EMG_instance = EMG_Signal.prepare(data,ElectrodeMappings,channel_df)
        self.fs = self.EMG_instance[0].fs
    
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
        EMG_Signal.segment(t_Epoch=3)
        test_epoch = EMG_Signal.Epoch[0]
        
        # mu, sigma = 0, 0.003
        # noise = np.random.normal(mu, sigma, test_epoch.shape)
        # test_epoch = test_epoch + noise
        
        test_epoch_shifted = Test.phaseShift(test_epoch)
        
        
        
        model = preprocessing.CCA(test_epoch,test_epoch_shifted,self.fs)
        model.fit()
        print(model.eigenvalues)
        decomp = model.decompose()
        result = model.reconstruct(decomp,0.92)
        EMG_Signal.visulize(result,t.fs,selected=[0],title='CCA',range=(0,0.2))
        EMG_Signal.visulize(test_epoch,t.fs,selected=[0],title='raw',range=(0,0.2))
        plt.show()
    
    def FilterTest(self,EMG_Signal):
        EMG_Signal.segment(t_Epoch=3)
        test_epoch = EMG_Signal.Epoch[1]
        #test_epoch_shifted = Test.phaseShift(test_epoch)
        
        filter1 = preprocessing.FIR_Filter(0)
        HP = filter1(40,20,self.fs,40,type='highpass')
        #preprocessing.FIR_Filter.BodePlot(HP,self.fs,title='HP')

        filter2 = preprocessing.FIR_Filter(0)
        LP = filter2(600,700,self.fs,40,type='lowpass')
        #preprocessing.FIR_Filter.BodePlot(LP,self.fs,title='LP')
        print(filter)
        
        for i in range(test_epoch.shape[0]):
            if i == 0:
                after = np.convolve(test_epoch[i],HP,mode='same')
            else:
                tempt = np.convolve(test_epoch[i],HP,mode='same')
                after = np.concatenate((after,tempt),axis=0)
        after = np.reshape(after,(test_epoch.shape[0],-1))
        
        
        for j in range(test_epoch.shape[0]):
            if j == 0:
                final = np.convolve(after[i],LP,mode='same')
            else:
                tempt = np.convolve(after[i],LP,mode='same')
                final = np.concatenate((final,tempt),axis=0)
        final = np.reshape(final,(test_epoch.shape[0],-1))
        
        
        #EMG_Signal.visulize(final,self.fs,selected=[2,3],title='final',range=(0,0.2))
        #EMG_Signal.visulize(after,self.fs,selected=[2,3],title='filtered')
        #EMG_Signal.visulize(test_epoch,self.fs,selected=[2,3],title='raw',range=(0,0.2))
        self.CCATest(final)
        plt.show()

    def NotchTest(self,EMG_Signal):
        EMG_Signal.segment(t_Epoch=3)
        test_epoch = EMG_Signal.Epoch[0]*1e3
        
        # b, a = signal.iirnotch(50, 30, fs=self.fs)       
        # y_notched = signal.filtfilt(b, a, test_epoch)
        y_notched = mne.filter.notch_filter(test_epoch,self.fs,50,method='fir')

        
        EMG_Signal.visulize(y_notched,self.fs,selected=[2,3],title='notch',range=(0,0.2))
        EMG_Signal.visulize(test_epoch,self.fs,selected=[2,3],title='raw',range=(0,0.2))
        
        plt.show()
    
    @staticmethod
    def phaseShift(data):
        new_data = np.zeros(data.shape)
        new_data[:,1:] = data[:,:-1]
        return new_data







if __name__ == '__main__':
    t = Test()
    r = t.CCATest(t.EMG_instance[0])
    #f = t.FilterTest(t.EMG_instance[0])
    #t.NotchTest(t.EMG_instance[0])