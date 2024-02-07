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
import preprocessing


class EMG_Signal():
    def __init__(self,data,f=1):
        
        '''

        initialize a signal instance
        
        Parameters:
        -----------
        data: ndarray
        matrix in shape (n_channel,n_sample)
        f: float
        sampling frequency of the data, default to 1 Hz 
        
        '''
        
        self.data = data
        self.freq = f
        try:
            self.n_channel = data.shape[0]
            self.n_sample = data.shape[1]
        except:
            self.n_channel = 1
            self.n_sample = data.shape[0]
        self.totalTime = (self.n_sample-1)/self.freq
        self.xTimeAxis = np.linspace(0,self.totalTime,self.n_sample)
        
    def zero_mean(self,axis=1):
        
        '''
        
        make the mean of input matrix along given axis to be 0
        
        Parameters:
        -----------
        dara : ndarray
        input matrix
        axis : 0 or 1, default to 1
        along rows or columns. 0 for coloumn and 1 for row.
        
        Returns:
        --------
        out : ndarray
        data array after zero mean
        
        '''
        if axis:
            for i in range(self.data.shape[0]):
                self.data[i,:] = self.data[i,:] - np.mean(self.data[i,:])
        else: 
            for j in range(self.data.shape[1]):
                self.data[:,j] = self.data[:,j] - np.mean(self.data[:,j])       
        return self.data
    
    def noise_generator(self,noise_type,multiple=1,range=None,**kwarg):
        
        '''
        
        Generate different noise based on current signals
        
        '''
        
        if noise_type == 'spikes':
            try:
                height = kwarg['height']
                width = kwarg['width']
                position = kwarg['position']
            except:
                raise TypeError ('Paramters for spike generation uncomplete')
            return self._spike_generator(height,width,position,multiple,range=range)
        elif noise_type == 'baseline':
            pass
        
        
    def _spike_generator(self,height,width,position,multiple,**kwarg):
        length = self.n_sample
        x = np.arange(0,length,1)/self.freq
        Random = False
        if position == 'random':
            Random = True
        if kwarg['range'] != None:
            _range = kwarg['range']
        else:
            _range = [min(x),max(x)]

        if width * multiple > max(x):
            raise ValueError ('Width too large')
        if Random:
            position = []
            i = 0
            count = 0
            while i<multiple:
                if count > multiple + 10:
                    raise Exception ('hard to randomize positions')
                accept = True
                tempt = random.uniform(_range[0],_range[1])
                for j in range(i):
                    if np.abs(tempt-position[j]) < width:
                        accept = False
                        break
                if accept:
                    position.append(tempt)
                    i += 1
                count += 1
            position = np.array(position)
        else:
            if len(position) != multiple:
                raise ValueError ('Number of positions incorrect')
            
        spike = np.zeros(length)
        for pos in position:    
            start = np.where(x<=pos-width/2)[0][-1]
            end = np.where(x>=pos+width/2)[0][0]
            l = end - start + 1
            if l < 1:
                raise ValueError ('width too small')
            l_len = int(l/2)
            r_len = l - l_len
            l_spike = np.linspace(0,height,num=l_len)
            r_spike = np.linspace(height,0,num=r_len)
            spike[start:start+l_len] = l_spike
            spike[start+l_len:end+1] = r_spike
        return spike


    def segment(self,**kwargs):
        
        '''
        
        Parameters:
        -----------
        t_Epoch : float
        period of an epoch in seconds
        n_Epoch : int
        number of epochs to be divided
         
        '''
        
        if 't_Epoch' in list(kwargs.keys()):
            self.t_Epoch = kwargs['t_Epoch']
            if self.t_Epoch > self.totalTime:
                raise ValueError ('Duration larger than total time')
            return self._segmentDuration()
        elif 'n_Epoch' in list(kwargs.keys()):
            self.n_Epoch = kwargs['n_Epoch']
        else:
            raise TypeError ('Unsupported')

    def _segmentDuration(self):
        self.n_Epoch = round(self.totalTime/self.t_Epoch)
        n_Epoch_sample = round(self.t_Epoch*self.freq)
        start = 0
        EpochSegmentation = []
        for i in range(self.n_Epoch):
            if start+n_Epoch_sample > self.n_sample:
                EpochSegmentation.append((start,self.n_sample))
            else:
                EpochSegmentation.append((start,start+n_Epoch_sample))
            start += n_Epoch_sample
        self.Epoch = []
        for index in EpochSegmentation:
            self.Epoch.append(self.data[:,index[0]:index[1]])
        return self.Epoch