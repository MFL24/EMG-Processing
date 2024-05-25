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
from scipy import signal
import mne
import time
from scipy.signal import kaiserord, lfilter, firwin, freqz

class Test(object):
    def __init__(self):
        filename = 'D:/Wenlong Li/TUM/Master/第一学期/FP/Data/Hao/2023-09-06_17h50_Hao_ramp_30MVC_and_sin/X_Iocz20230906173031_03_Hao_ramp_30MVC_and_sin.otb+'
        data, channel_df = data_readers.import_otb(filename)
        self.EMG_instance = EMG_Signal.prepare(data,ElectrodeMappings,channel_df)
        self.fs = self.EMG_instance[0].fs
        print('fs = {}'.format(self.fs))
    
    def OutlinerTest(self,EMG_Signal):
        EMG_Signal.segment(t_Epoch=3)
        test_epoch = EMG_Signal.Epoch[100]

        t1 = time.time()
        p = preprocessing.Outliner_detection(test_epoch,EMG_Signal.metadata,0.5)
        p.outliner_detection()  
        p.reference_selection()
        p.threshold_calculation(1,0.8,2)
        a = p.detect()
        t2 = time.time()
        print(t2-t1)
        p.OutlinerVisulize()
        #p.visulize(p.data,p.fs,selected=[5,6,4,12,13,14])
        plt.show()        

    def DetrendTest(self,EMG_Signal,method='median',**kwargs):
        EMG_Signal.segment(t_Epoch=3)
        test_epoch = EMG_Signal.Epoch[20]

        model = preprocessing.Detrend(test_epoch,method=method,**kwargs)
        baseline,residual = model.apply()
        # print(baseline[2,:].mean())
        # print(residual[2,:].mean())
        # print(test_epoch[2,:].mean())
        model.visulize(baseline,EMG_Signal.fs,selected=[2,3],title='baseline')
        model.visulize(residual,EMG_Signal.fs,selected=[2,3],title='residual')
        model.visulize(test_epoch,EMG_Signal.fs,selected=[2,3],title='raw')
        plt.show()
    
    def CCATest(self,EMG_Signal):
        EMG_Signal.segment(t_Epoch=3)
        test_epoch = EMG_Signal.Epoch[100]['data']
        
        mu, sigma = 0, 0.05
        noise = np.random.normal(mu, sigma, test_epoch.shape)
        test_epoch = test_epoch + noise
        
        test_epoch_shifted = Test.phaseShift(test_epoch)
        
        
        t1 = time.time()
        model = preprocessing.CCA(test_epoch,test_epoch_shifted,self.fs)
        model.fit()
        #print(model.eigenvalues)
        decomp = model.decompose()
        result = model.reconstruct(decomp,0.9)
        t2 = time.time()
        # T = 1/self.fs
        # x = np.arange(0,test_epoch.shape[1])*T 
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.plot(x,test_epoch[2,:],label='raw')
        # ax.plot(x,result[2,:],label='CCA')
        # ax.legend()
        print(t2-t1)
        EMG_Signal.visulize(result,t.fs,selected=[0],title='CCA',range=(1,1.2))
        EMG_Signal.visulize(test_epoch,t.fs,selected=[0],title='raw',range=(1,1.2))
        print('raw RMS: {}'.format(FeatureExtraction.T_Domain.RMS(test_epoch,mean=True)))
        print('final RMS: {}'.format(FeatureExtraction.T_Domain.RMS(result,mean=True))) 
        plt.show()
    
    def FilterTest(self,EMG_Signal):
        EMG_Signal.segment(t_Epoch=3)
        test_epoch = EMG_Signal.Epoch[1]
        #test_epoch_shifted = Test.phaseShift(test_epoch)
        
        filter1 = preprocessing.FIR_Filter(0)
        HP = filter1(10,5,self.fs,40,type='highpass')
        #preprocessing.FIR_Filter.BodePlot(HP,self.fs,title='HP')

        filter2 = preprocessing.FIR_Filter(0)
        LP = filter2(700,750,self.fs,40,type='lowpass')
        #preprocessing.FIR_Filter.BodePlot(LP,self.fs,title='LP')
        print(len(LP))
        print(len(HP))
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.plot(LP, 'bo-')
        t1 = time.time()
        # for i in range(test_epoch.shape[0]):
        #     if i == 0:
        #         after = np.convolve(test_epoch[i],HP,mode='same')
        #     else:
        #         tempt = np.convolve(test_epoch[i],HP,mode='same')
        #         after = np.concatenate((after,tempt),axis=0)
        # after = np.reshape(after,(test_epoch.shape[0],-1))
        
        after = test_epoch
        
        for j in range(test_epoch.shape[0]):
            if j == 0:
                final = np.convolve(after[j],LP,mode='same')
            else:
                tempt = np.convolve(after[j],LP,mode='same')
                final = np.concatenate((final,tempt),axis=0)
        final = np.reshape(final,(test_epoch.shape[0],-1))
        
        #final = np.apply_along_axis(lambda y: np.convolve(y, LP,mode='same'), 0, test_epoch)
        #final = np.apply_along_axis(lambda y: np.convolve(y, HP,mode='same'), 0, test_epoch)
        # final = lfilter(LP, 1.0, test_epoch,axis=0)
        # final = lfilter(HP, 1.0, final,axis=0)
        
        t2 = time.time()
        print(t2-t1)
        EMG_Signal.visulize(final,self.fs,selected=[2,3],title='final',range=(0,0.2))
        # #EMG_Signal.visulize(after,self.fs,selected=[2,3],title='filtered')
        EMG_Signal.visulize(test_epoch,self.fs,selected=[2,3],title='raw',range=(0,0.2))

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
    
    def zscoreTest(self,Signal):
        Signal.segment(t_Epoch=3)
        test_epoch = EMG_Signal(Signal.Epoch[2],Signal.metadata)
        spike = test_epoch.noise_generator('spikes',2,height=0.5,width=0.05,position=[0.23,2.2])
        data = test_epoch.data + spike
        t1 = time.time()
        algo = preprocessing.Z_Score_thresholding(data,Signal.metadata,0.5)
        algo.apply(5)
        t2 = time.time()
        print(t2-t1)
        EMG_Signal.visulize(data,self.fs,selected=[2,3],title='spike')
        plt.show()

    def finalTest(self,Signal):
        Signal.segment(t_Epoch=3)
        test_epoch = EMG_Signal(Signal.Epoch[2]['data'],Signal.metadata)
        spike = test_epoch.noise_generator('spikes',2,height=0.5,width=0.05,position=[0.23,2.2])
        data = test_epoch.data + spike
        mu, sigma = 0, 0.05
        noise = np.random.normal(mu, sigma, data.shape)
        raw_data = data + noise
        m1 = FeatureExtraction.T_Domain.RMS(data,mean=True)
        m2 = FeatureExtraction.T_Domain.RMS(noise,mean=True)
        SNR  = 20*np.log(m1/m2)
        print('SNR: {}dB'.format(SNR))
        p = preprocessing.Outliner_detection(data,Signal.metadata,0.5)
        p.outliner_detection()  
        p.reference_selection()
        p.threshold_calculation(1,2.5,15)
        a = p.detect()
        final = p.clip()

        p.OutlinerVisulize()
        algo = preprocessing.Z_Score_thresholding(final,Signal.metadata,0.5)
        algo.apply(5.2)        
        final = algo.clip()      
          
        data_shifted = Test.phaseShift(final)
        
        t1 = time.time()
        model = preprocessing.CCA(final,data_shifted,self.fs)
        model.fit()
        decomp = model.decompose()
        result = model.reconstruct(decomp,0.9)     
        
        # print('RAW')
        # algo = preprocessing.Z_Score_thresholding(data,Signal.metadata,0.5)
        # algo.apply(5)           
        # print('after CCA')


        t2 = time.time()        
        # EMG_Signal.visulize(result,self.fs,title='final',selected=[2,3],bad_channel_list=a['bad_ch'])

        # EMG_Signal.visulize(data,self.fs,title='raw',selected=[2,3],bad_channel_list=a['bad_ch'])        
        
        fig = plt.figure()
        ax = fig.add_subplot()
        
        ax.plot(raw_data[2,:],label='raw')
        ax.plot(result[2,:],label='final')
        ax.legend()
        
        
        
        
        plt.show()
        
    @staticmethod
    def phaseShift(data):
        new_data = np.zeros(data.shape)
        new_data[:,1:] = data[:,:-1]
        return new_data







if __name__ == '__main__':
    t = Test()
    #t.OutlinerTest(t.EMG_instance[0])
    #r = t.CCATest(t.EMG_instance[0])
    #f = t.FilterTest(t.EMG_instance[0])
    #t.NotchTest(t.EMG_instance[0])
    #t.DetrendTest(t.EMG_instance[0],r=20)
    #t.DetrendTest(t.EMG_instance[0],method='qvr',lamda=5)
    #t.zscoreTest(t.EMG_instance[0])
    t.finalTest(t.EMG_instance[0])
    