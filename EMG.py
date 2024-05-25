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
import Plot_toolbox
from otb_matrices import otb_patch_map
from ElectrodeMapping import *

class EMG_Signal():
    
    @classmethod
    def prepare(cls,data,ElectrodeMapping,metadata):
        
        '''
        
        Croping the initial data into parts representing individual arrays
        and decode the metadata.
        
        Parameters:
        -----------
        data : np.ndarray
        the initial data comprising of different arrays and other sensors
        ElectrodeMapping : dict
        map stating the positions of electrode with keys as sensortype
        metadata : dict
        containg details of data 
        
        Returns:
        --------
        tempt_instance : list of EMG_Signal instance
        every element of tempt_instance is an EMG_Signal instance 
        established from every single array 
        
        '''
        
        cls.data = data
        cls.ElectrodeMapping = ElectrodeMapping
        cls.metadata = metadata
        cls._DecodeMetadata()
        cls.ArrayIndex = []
        ArrayNumber = 1
        tempt_metadata = []
        
        for i in range(data.shape[0]):
            if cls.get_arrayNum(i) == 0:
                cls.ArrayIndex.append(i)        
                break
            elif cls.get_arrayNum(i) == ArrayNumber:
                tempt_M = {}
                cls.ArrayIndex.append(i)
                tempt_M['muscle'] = cls.get_muscle(i)
                tempt_M['sensor'] = cls.get_sensor(i)
                tempt_M['arrayNum'] = ArrayNumber
                tempt_M['emap'] = cls.ElectrodeMapping[tempt_M['sensor']]["ElChannelMap"]
                tempt_M['fs'] = cls.get_fsamp(i)
                ArrayNumber += 1
                tempt_metadata.append(tempt_M)

        tempt_data = []
        for j in range(len(cls.ArrayIndex)):
            if j+1 < len(cls.ArrayIndex):
                tempt_data.append(cls.data[cls.ArrayIndex[j]:cls.ArrayIndex[j+1]])
            else: 
                continue
            
        temtp_Instance = []
        for index in range(len(tempt_data)):
            temtp_Instance.append(EMG_Signal(tempt_data[index],tempt_metadata[index]))
        return temtp_Instance

    @classmethod
    def _DecodeMetadata(cls):
        cls.fs = cls.metadata['fsamp'].to_numpy()
        cls.muscle = cls.metadata['muscle'].to_numpy()
        cls.LP_filter = cls.metadata['LP_filter'][0]
        cls.HP_filter = cls.metadata['HP_filter'][0]
        cls.array_num = cls.metadata['array_num'].to_numpy()
        tempt = cls.array_num != cls.array_num
        cls.array_num[tempt] = 0
        cls.sensor = cls.metadata['sensor'].to_numpy()        
        cls.rel_index = cls.metadata['rel_index'].to_numpy()    
    
    @classmethod
    def get_arrayNum(cls,ch_num):
        return cls.array_num[ch_num]
    
    @classmethod
    def get_muscle(cls,ch_num):
        return cls.muscle[ch_num]

    @classmethod
    def get_sensor(cls,ch_num):
        return cls.sensor[ch_num]
        
    @classmethod
    def get_arrayNum(cls,ch_num):
        return cls.array_num[ch_num]
    
    @classmethod
    def get_relIndex(cls,ch_num):
        return cls.rel_index[ch_num]
    
    @classmethod
    def get_position(cls,ch_num):
        sensor = cls.get_sensor(ch_num)
        if not sensor in cls.sensortype:
            return 0
        else:
            return np.where(cls.ElectrodeMapping[sensor]==cls.get_relIndex(ch_num)+1)[0]   
    
    @classmethod
    def get_fsamp(cls,ch_num):
        return cls.fs[ch_num]
    
         
    def __init__(self,data,metadata):
        
        '''

        initialize an EMG_Signal instance
        
        Parameters:
        -----------
        data: ndarray
        matrix in shape (n_channel,n_sample)
        metadata: dict
        dict with keys as 'sensor', 'muscle', 'arrayNum', 'emap', 'fs',
        representing details of the data array
        
        '''
        
        self.data = data
        self.metadata = metadata
        self.sensortype = self.metadata['sensor']
        self.muscle = self.metadata['muscle']
        self.array_num = self.metadata['arrayNum']
        self.EMap = self.metadata['emap']
        self.fs = self.metadata['fs']
        self.ArrayInfo = self.EMap.shape
        try:
            self.n_channel = self.data.shape[0]
            self.n_sample = self.data.shape[1]
        except:
            self.n_channel = 1
            self.n_sample = self.data.shape[0]
            
        self.EPosition()
        self.Neighbour()
        self.totalTime = (self.n_sample-1)/self.fs
        self.xTimeAxis = np.linspace(0,self.totalTime,self.n_sample)
        self.ch_info = {'good_ch' : np.arange(self.n_channel),
                        'bad_ch' : None,
                        'rLFP' : None,
                        'rPLIP' : None,
                        'rms' : None}
        
    def EPosition(self):
        self.position = []
        for ch in range(self.n_channel):
            self.position.append(self.get_position(ch))
    
    def Neighbour(self):
        self.neighbour = []
        for ch in range(self.n_channel):
            self.neighbour.append(self.get_neigbour(ch))
        
    def get_position(self,ch_num):
        tempt = np.where(self.EMap==ch_num+1)
        return (tempt[0][0],tempt[1][0]) 
    
    def get_electrode(self,position):
        return self.EMap[position]-1
    
    def get_neigbour(self,ch_num):
        max_row , max_col = self.ArrayInfo
        pos = self.position[ch_num]
        row , col = pos
        tempt_dict = {}
        
        if col == 0 and row == 0:
            tempt_dict['Other'] = [self.get_electrode((row+1,col+1))]
        elif col == 0 and row == max_row-1:
            tempt_dict['Other'] = [self.get_electrode((row-1,col+1))]    
        elif col == max_col-1 and row == 0:
            tempt_dict['Other'] = [self.get_electrode((row+1,col-1))]
        elif col == max_col-1 and row == max_row-1:
            tempt_dict['Other'] = [self.get_electrode((row-1,col-1))]
            
        if col == 0:
            tempt_dict['E_W'] = [self.get_electrode((row,col+1))]
            if row not in [0,max_row-1]:
                tempt_dict['Other'] = [self.get_electrode((row+1,col+1)),
                                       self.get_electrode((row-1,col+1))]
        elif col == max_col-1:
            tempt_dict['E_W']= [self.get_electrode((row,col-1))]
            if row not in [0,max_row-1]:
                tempt_dict['Other'] = [self.get_electrode((row-1,col-1)),
                                       self.get_electrode((row+1,col-1))]            
        else:
            tempt_dict['E_W']= [self.get_electrode((row,col-1)),
                                self.get_electrode((row,col+1))]
        if row == 0:
            tempt_dict['N_S'] = [self.get_electrode((row+1,col))]
            if col not in [0,max_col-1]:
                tempt_dict['Other'] = [self.get_electrode((row+1,col-1)),
                                       self.get_electrode((row+1,col+1))]            
        elif row == max_row-1:
            tempt_dict['N_S']= [self.get_electrode((row-1,col))]
            if col not in [0,max_col-1]:
                tempt_dict['Other'] = [self.get_electrode((row-1,col-1)),
                                       self.get_electrode((row-1,col+1))]  
        else:
            tempt_dict['N_S']= [self.get_electrode((row-1,col)),
                                self.get_electrode((row+1,col))]                
        
        if not 'Other' in tempt_dict.keys():
            tempt_dict['Other'] = [self.get_electrode((row+1,col-1)),
                                   self.get_electrode((row+1,col+1)), 
                                   self.get_electrode((row-1,col+1)),
                                   self.get_electrode((row-1,col-1))]
        return tempt_dict     
                                  
                                  
    
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
        x = np.arange(0,length,1)/self.fs
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
        
        Return:
        Epoch : list of data array
        
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
        n_Epoch_sample = round(self.t_Epoch*self.fs)
        start = 0
        self.Epoch = []
        epoch_index = 0
        for _ in range(self.n_Epoch):
            tempt_epoch = {}
            if start+n_Epoch_sample > self.n_sample:
                tempt_epoch['data'] = self.data[:,start:]
                tempt_epoch['start'] = start
                tempt_epoch['end'] = self.n_sample
                self.Epoch.append(tempt_epoch)
            else:
                tempt_epoch['data'] = self.data[:,start:start+n_Epoch_sample]
                tempt_epoch['start'] = start
                tempt_epoch['end'] = start+n_Epoch_sample
                self.Epoch.append(tempt_epoch)
            start += n_Epoch_sample
            epoch_index += 1
        return self.Epoch
    
    @staticmethod
    def visulize(data,fs,ch_name='default',selected='all',**kwargs):
        if ch_name == 'default':
            ch_name = ['{}'.format(i) for i in range(data.shape[0])]
        else:
            ch_name = ch_name
        
        if selected == 'all':
            new_data = data
            new_ch_name = ch_name
        else:
            new_ch_name = []
            for index,i in enumerate(selected):
                if index == 0:
                    new_data = data[i,:]
                else:
                    new_data = np.concatenate((new_data,data[i,:]),axis=0)
                new_ch_name.append(ch_name[i])
            new_data = np.reshape(new_data,(len(selected),-1))
        
        if new_data.shape[0] != len(new_ch_name):
            raise ValueError ('mismatch between channel number')
        fig = Plot_toolbox.MultiChannelPlot()        
        fig.plot(fs,new_data,new_ch_name,**kwargs)
        
    