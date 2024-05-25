from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.linalg import inv
import sys
import os
import random
import scipy.signal
from scipy import integrate
from EMG import EMG_Signal
import preprocessing


class F_Domain(EMG_Signal):
    
    def PSD(self):
        
        '''
        
        Parameters:
        -----------
        data : ndarray
        input data with rows as channels
        
        Return:
        -------
        f : ndarray
        frequency axis after PSD
        P : ndarray
        psd values
         
        '''
        
        for row in range(self.data.shape[0]):
            f_tempt,P_tempt = scipy.signal.welch(self.data[row,:],fs=self.fs,nperseg=self.fs)
            if row == 0:
                f = f_tempt
                P = P_tempt
            else:
                f = np.concatenate((f,f_tempt),axis=0)
                P = np.concatenate((P,P_tempt),axis=0)
        f = np.reshape(f,(self.data.shape[0],-1))
        P = np.reshape(P,(self.data.shape[0],-1))
        return [f,P]    


class T_Domain(EMG_Signal):
    
    @staticmethod
    def RMS(data,mean=False,axis=0):
        if data.ndim == 1:
            rms = np.sqrt(np.sum(data**2)/len(data))
        else:
            rms = []   
            for i in range(data.shape[axis]):
                if axis == 0:
                    rms.append(T_Domain.RMS(data[i,:]))
                else:
                    rms.append(T_Domain.RMS(data[:,i]))
        return np.mean(rms) if mean else np.array(rms)
    
    @staticmethod
    def Mean_and_Variance(data):
        if data.ndim == 1:
            return (np.mean(data),np.std(data))
        else:
            mean = []
            std = []
            for i in range(data.shape[0]):
                sig = data[i,:]
                mean.append(np.mean(sig))
                std.append(np.std(sig))
            return (mean,std)
    
    @staticmethod
    def Z_Score(data):
        mean,std = T_Domain.Mean_and_Variance(data)
        if data.ndim == 1:
            zScore = np.zeros(len(data))
            for i in range(len(data)):
                zScore[i] = (data[i]-mean)/std
            return zScore
        else:
            zScore = np.zeros(data.shape[1])
            for row in range(data.shape[0]):
                sig = data[row,:]
                zScore = zScore + T_Domain.Z_Score(sig)
            return zScore/data.shape[0]

                
                