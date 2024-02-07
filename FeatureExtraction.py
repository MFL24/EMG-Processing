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
from __init__ import * 
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
            f_tempt,P_tempt = scipy.signal.welch(self.data[row,:],fs=self.freq)
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
    def RMS(self):
        try:
            n_channel = self.data.shape[0]
            n_sample = self.data.shape[1]
        except:
            n_channel = 1
            n_sample = self.data.shape[0]    
        rms = []   
        for i in range(n_channel):
            rms.append(np.sqrt(np.sum(self.data[i,:]**2)/n_sample))
        return np.array(rms) 