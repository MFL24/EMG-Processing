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
import FeatureExtraction

class FIR_Filter():
    def __init__(self,function):
        self.function = function
                
    def __call__(self,fp,fs,ripple,type='lowpass'):
        if type == 'lowpass':
            if fs < fp:
                raise ValueError ('fs smaller than fp')
            return self._LowPass(fp,fs,ripple)
        if type == 'highpass':
            if fs > fp:
                raise ValueError ('fp smaller than fs')
            return self._HighPass(fp,fs,ripple)
    
    @property
    def WindowFunc(self):
        return self._window_func
    
    @property
    def IdealFunc(self):
        return self._ideal_func   
    
    @WindowFunc.setter
    def WindowFunc(self,w):
        #if isinstance(func,ndarray):
        self._window_func = w
    
    @IdealFunc.setter
    def IdealFunc(self,h):
        #if isinstance(func,ndarray):
        self._ideal_func = h   
    
    def _LowPass(self,fp,fs,ripple):
        fc = (fp+fs)/2
        self.WindowFunc = self._getWindow(fp,fs,ripple)
        self.IdealFunc = self._getIdealFilter(fc,type='lowpass')
        FinalFilter = self.extendFilter(self.WindowFunc*self.IdealFunc)
        return FinalFilter
    
    def _HighPass(self,fp,fs,ripple):
        fc = (fp+fs)/2
        self.WindowFunc = self._getWindow(fp,fs,ripple)
        self.IdealFunc = self._getIdealFilter(fc,type='highpass')
        FinalFilter = self.extendFilter(self.WindowFunc*self.IdealFunc)
        return FinalFilter
        
    def _getWindow(self,fp,fs,ripple):
        transitionBandwidth = np.abs(fs-fp)
        if ripple < 25:
            self.order = round(2/transitionBandwidth -1)
            return self._RectWindow()
        elif 24 < ripple < 44:
            self.order = round(4/transitionBandwidth)
            return self._HannWindow()
        elif 43 < ripple < 53:
            self.order = round(4/transitionBandwidth)
            return self._HanningWindow()
        elif 52 < ripple < 74:
            self.order = round(6/transitionBandwidth)
            return self._BlackmannWindow()
        else:
            raise ValueError ('ripple too large')
    
    def _RectWindow(self):
        return np.ones(self.order+1)
    
    def _HannWindow(self):
        w = []
        for n in range(self.order+1):
            w.append(0.5-0.5*np.cos(2*n*np.pi/self.order))
        return np.array(w)
    
    def _HanningWindow(self):
        w = []
        for n in range(self.order+1):
            w.append(0.54-0.46*np.cos(2*n*np.pi/self.order))
        return np.array(w)        
    
    def _BlackmannWindow(self):
        w = []
        for n in range(self.order+1):
            w.append(0.42-0.5*np.cos(2*n*np.pi/self.order)+0.08*np.cos(4*n*np.pi/self.order))
        return np.array(w)           
    
    def _getIdealFilter(self,fc,type='lowpass'):
        wc = 2*np.pi*fc
        tempt = wc/np.pi
        h = []
        tt = int(self.order/2)
        for n in range(-tt,tt+1):
            if type == 'lowpass':
                h.append(tempt*np.sinc(tempt*n))
            elif type == 'highpass':
                h.append(np.sinc(n)-(tempt*np.sinc(tempt*n)))
        return np.array(h)

    def extendFilter(self,filter,factor=5):
        new_filter = np.zeros(factor*len(filter))
        left = int((factor-1)/2)
        new_filter[left*len(filter):(left+1)*len(filter)] = filter
        return new_filter
    
    @staticmethod
    def BodePlot(transfer_func):
        a = np.fft.rfft(transfer_func)
        norm = np.abs(a)
        norm_log = 20 * np.log10(norm)
        #phase = np.angle(a)
        freq = np.fft.rfftfreq(len(transfer_func))
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(freq,norm_log,label='norm')
        #ax.plot(freq,phase,label='phase')
        ax.legend()


class Preprocessing(EMG_Signal):
    def detrend(self,method,**kwarg):
        
        '''
        
        Remove the baseline artefact 
        
        Parameters:
        -----------
        method : 'median' or 'qvr'
        specific method used for detrending. 'median' for median filter and 'qvr' for quadractic variation reduction
        r : int
        only in 'median' method. The filter window radius.
        lamda : float
        only in 'qvr'. Controls the precision of the baseline detected. 
        
        Returns:
        -------
        out : list 
        first element is the baseline, second is the denoised signal. 
        
        '''
        
        if method == 'median':
            r = kwarg['r']
            return self._median_filter(r)
        elif method == 'qvr':
            lamda = kwarg['lamda']
            self.D = np.zeros((self.n_sample-1,self.n_sample))
            for i in range(self.n_sample-1):
                self.D[i,i] = 1
                self.D[i,i+1] = 1
            return self._QuadracticVariationReduction(lamda)
        else:
            raise ValueError ('method not supported')
           
    def _median_filter(self,r):
        data = self.data
        l = len(data)
        baseline = [] 
        for i in range(l):
            if i < r:
                interval = np.zeros(2*r+1)
                interval[r-i:] = data[:r+i]
                m = np.median(interval)
                baseline.append(m)
            elif i + r > l:
                interval = np.zeros(2*r+1)
                interval[:l-i+r] = data[i-r:]
                baseline.append(m)
            else:
                interval = data[i-r:i+r+1]
                baseline.append(m)
        return [baseline,data-baseline]

    def _QuadracticVariationReduction(self,lamda):
        baseline_matrix = inv(np.eye(self.n_sample)+lamda*np.matmul(self.D.T,self.D))
        self.baseline = np.matmul(baseline_matrix,self.data.T)
        self.baseline = self.baseline.T
        return [self.baseline,self.data - self.baseline]


    def whitening(self,method='PCA',regularization=0):
        
        '''
        
        Whitening the data matrix
        
        Parameters:
        -----------
        method: 'PCA' or 'ZCA'
        method to compute the whitening
        
        Returns:
        --------
        data: ndarray
        whitened data
        
        
        '''
        
        data = self.zero_mean(self.data)
        covriance_matrix = np.matmul(self.data,np.transpose(self.data))
        eigenvalues, eigenvectors = LA.eig(covriance_matrix)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        transform_matrix = eigenvectors[:,idx] 
        
        L = np.diag(eigenvalues+regularization)
        L = int(L)**0.5
        data = np.matmul(transform_matrix.T,data)
        data = np.matmul(L,data)
        if method == 'PCA':
            return data
        elif method == 'ZCA':
            return np.matmul(transform_matrix,data)
        else:
            raise TypeError ('Invalid method')
        
        

class Outliner_detection(EMG_Signal):
    def outliner_detection(self,T):
        epochs = self.segment(t_epoch = T)
        relativeLowFreqPower = []
        relativePLIPower = []
        rms = []
        for epoch in epochs:
            freq_Epoch,PSD_Epoch = FeatureExtraction.F_Domain(epoch,self.freq).PSD()
            relativeLowFreqPower.append(self._relativeLowFreqComponentPower(freq_Epoch,PSD_Epoch))
            relativePLIPower.append(self._relativePLIPower(freq_Epoch,PSD_Epoch))
            rms.append(FeatureExtraction.T_Domain(epoch,self.freq).RMS())
    
    def _average_feature(data):    
        for i in data:
            if i == data[0]:
                sum = i
            else:
                sum = sum + i
        return sum/len(data)
        
        
    def _relativeLowFreqComponentPower(freq,Power,max_threshold_freq=500,low_threshold_freq=12):
        n_ch = freq.shape[0]
        relative_low_freq_power = []
        for ch in range(n_ch):
            f = freq[ch,:]
            p = Power[ch,:]
            max_index = np.where(f<max_threshold_freq)[0][-1]
            low_index = np.where(f<low_threshold_freq)[0][-1]
            power_full = integrate.simpson(p[:max_index],f[:max_index])
            power_low = integrate.simpson(p[:low_index],f[:low_index])
            relative_low_freq_power.append(power_low/power_full)
        return np.array(relative_low_freq_power)
        
    def _relativePLIPower(self,freq,Power,max_threshold_freq=500,n_harmonics=4):
        PLI_freq = []
        for i in range(n_harmonics):
            PLI_freq.append(50*(i+1))
        n_ch = freq.shape[0]
        power_PLI = 0
        relativePLIPower = []
        for ch in range(n_ch):
            f = freq[ch,:]
            p = Power[ch,:]
            max_index = np.where(f<max_threshold_freq)[0][-1]
            power_full = integrate.simpson(p[:max_index],f[:max_index])
            PLI_index = []
            for PLI_f in PLI_freq:
                PLI_index.append(np.where(f<PLI_f)[0][-1]) 
            for index in PLI_index:
                power_PLI = power_PLI + Power[index]
            relativePLIPower.append(power_PLI/power_full)
        return np.array(relativePLIPower)
    


class PCA2(EMG_Signal):
    
    '''
    
    compute PCA algorithm on given data
    
    
    '''
    
    def __init__(self,data):
        self.data = data
        self.nrow = data.shape[0]
        self.ncol = data.shape[1]
        self.U = None
        self.S = None
        self.V = None
        self.eigenvalues = None
        self.transform_matrix = None
        self.n_component = None
        
    def fit(self,method='EVD'):
        self.data = self.zero_mean(self.data)
        
        if method == 'EVD':
            return self._fit_evd()
        elif method == 'SVD':
            return self._fit_svd()
        else:
            raise TypeError ('Method not supported')
        
    def _fit_evd(self):
        covriance_matrix = np.matmul(self.data,np.transpose(self.data))
        eigenvalues, eigenvectors = LA.eig(covriance_matrix)
    
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.transform_matrix = eigenvectors[:,idx]        
        
    def _fit_svd(self):
        U, S, Vt = LA.svd(self.data)
        self.U = U
        self.S = S
        self.V = Vt.T
        
    def compute(self,n_component,direction='row'):
        if direction == 'row':
            if n_component > self.nrow:
                raise ValueError ('number of components too large')
            self.n_component = n_component
            if not self.eigenvalues is None:
                return self._compute_evd()
            else:
                return self._compute_svd_row()
        elif direction == 'col':
            if n_component > self.ncol:
                raise ValueError ('number of components too large')
            self.n_component = n_component
            if not self.eigenvalues is None:
                raise TypeError ('EVD method not supported for col decomposition')
            else:
                return self._compute_svd_col()            
        else:
            raise ValueError ('unknown direction')
        
    def _compute_evd(self):
        decomposition_matrix = self.transform_matrix.T[:self.n_component,:]
        return np.matmul(decomposition_matrix,self.data)
    
    def _compute_svd_row(self):
        decomposition_matrix = self.U.T[:self.n_component,:]
        return np.matmul(decomposition_matrix,self.data)
    
    def _compute_svd_col(self):
        decomposition_matrix = self.V[:,:self.n_component]
        return np.matmul(self.data,decomposition_matrix)
    
    def reconstruct(self):
        if self.n_component is None:
            raise TypeError ('compute with svd method must be called first')
        else:
            sigma = np.zeros((self.nrow,self.ncol))
            for i in range(self.nrow):
                sigma[i,i] = self.S[i]
            
            tempt = np.matmul(self.U[:,:self.n_component],sigma[:self.n_component,:])
            return np.matmul(tempt,self.V.T)



class CCA(EMG_Signal):
    
    def __init__(self,X,Y,freq):
        if not X.shape == Y.shape:
            raise ValueError ('X and Y must have same dimension')    
        self.X = X 
        self.Y = Y   
        self.nrow = X.shape[0]
        self.freq = freq
        
    def fit(self,ignoreY=True,alpha=1):
        sigma_XX = np.matmul(self.X,self.X.T)
        sigma_YY = np.matmul(self.Y,self.Y.T)
        sigma_XY = np.matmul(self.X,self.Y.T)
        sigma_YX = np.matmul(self.Y,self.X.T)
        
        A = inv(sigma_XX) @ sigma_XY
        A = A @ inv(sigma_YY)
        A = A @ sigma_YX
        
        eigenvalues, eigenvectors = LA.eig(A)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        transform_matrix_X = eigenvectors[:,idx]     

        self.transfirmation_matrix_X = transform_matrix_X.T
        self.eigenvalues = eigenvalues      
        
        if not ignoreY:
            B = inv(sigma_YY) @ sigma_YX
            transform_matrix_Y = B @ transform_matrix_X
            transform_matrix_Y *= alpha
            self.transfirmation_matrix_Y = transform_matrix_Y.T
            
    def decompose(self):
        if not hasattr(self,'transfirmation_matrix_X'):
            raise TypeError ('must call fit first')
        elif hasattr(self,'transfirmation_matrix_Y'):
            X_decompose = np.matmul(self.transfirmation_matrix_X,self.X)
            Y_decompose = np.matmul(self.transfirmation_matrix_Y,self.Y)
            return [X_decompose,Y_decompose]
        else:
            X_decompose = np.matmul(self.transfirmation_matrix_X,self.X)
            return X_decompose
    
    def reconstruct(self,data,threshold,type='X',criterion='correlation coefficient'):
        if criterion == 'correlation coefficient':
            if threshold < self.eigenvalues.min():
                raise ValueError ('Threshold larger than biggest ')
            self.threshold = threshold
            self.n_component = np.where(self.eigenvalues>threshold)[0][-1] + 1
            source = data
            for i in range(self.n_component,data.shape[0]):
                source[i,:] = np.zeros(data.shape[1])
                
            if type == 'X':
                return np.matmul(inv(self.transfirmation_matrix_X),source)
            elif type == 'Y':
                return np.matmul(inv(self.transfirmation_matrix_Y),source)
            else:
                raise TypeError ('type must be X or Y')
