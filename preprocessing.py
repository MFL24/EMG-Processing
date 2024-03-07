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
from otb_matrices import otb_patch_map
from matplotlib.patches import Wedge
import matplotlib.patches as mpatches


class FIR_Filter():
    def __init__(self,function):
        self.function = function
                
    def __call__(self,fp,fs,f_samp,ripple,type='lowpass'):
        if type == 'lowpass':
            if fs < fp:
                raise ValueError ('fs smaller than fp')
            return self._LowPass(fp/f_samp,fs/f_samp,ripple)
        if type == 'highpass':
            if fs > fp:
                raise ValueError ('fp smaller than fs')
            return self._HighPass(fp/f_samp,fs/f_samp,ripple)
    
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
            if self.order % 2 != 0:
                self.order += 1
            return self._RectWindow()
        elif 24 < ripple < 44:
            self.order = round(4/transitionBandwidth)
            if self.order % 2 != 0:
                self.order += 1
            return self._HannWindow()
        elif 43 < ripple < 53:
            self.order = round(4/transitionBandwidth)
            if self.order % 2 != 0:
                self.order += 1
            return self._HanningWindow()
        elif 52 < ripple < 74:
            if self.order % 2 != 0:
                self.order += 1
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
    def BodePlot(transfer_func,fs,**kwargs):
        a = np.fft.rfft(transfer_func)
        norm = np.abs(a)
        norm_log = 20 * np.log10(norm)
        #phase = np.angle(a)
        freq = np.fft.rfftfreq(len(transfer_func))
        fig = plt.figure()
        ax = fig.add_subplot()
        if 'title' in kwargs:
            ax.set_title(kwargs['title'])
        ax.plot(freq*fs,norm_log,label='norm')
        #ax2 = ax.twinx()
        #ax.plot(freq*fs,phase,label='phase')
        ax.legend()


class Detrend(EMG_Signal):
    
    '''
    
    detrend corresponding data to find the residual or baseline of the signal
    Two methods ('median' and 'qvr') are supported.
    median method is relatively faster most likely
    
    '''
    
    def __init__(self,data,method='median',**kwargs):
        
        '''
        
        Initialize detrend operation
        
        Parameters:
        -----------
        method : 'median' or 'qvr'
        methods to caculate baseline
        r : int
        the radius of the median filter in 'median' method
        lamda : float
        only in 'qvr'. Controls the precision of the baseline detected.          
        
        '''
        
        self.data = data
        self.n_channel = data.shape[0]
        self.n_sample = data.shape[1]
        if method == 'median':
            self.method = 'median'
            try:
                self.r = kwargs['r']
            except:
                raise KeyError ('r argument required')
        elif method == 'qvr':
            self.method = 'qvr'
            try:
                self.lamda = kwargs['lamda']
            except:
                raise KeyError ('lamda argument required')        
    
    def apply(self,**kwarg):
        
        '''
        
        Calculate the baseline according to given method 
        
        Returns:
        -------
        out : list 
        first element is the baseline, second is the residual. 
        
        '''
        
        if self.method == 'median':
            for ch in range(self.n_channel):
                tempt_bassline,tempt_residual = self._median_filter(self.data[ch])
                if ch == 0:
                    baseline = tempt_bassline
                    residual = tempt_residual
                else:
                    baseline = np.concatenate((baseline,tempt_bassline),axis=0)
                    residual = np.concatenate((residual,tempt_residual),axis=0)
            baseline = np.reshape(baseline,(self.n_channel,-1))
            residual = np.reshape(residual,(self.n_channel,-1))
            return (baseline,residual)
        elif self.method == 'qvr':
            self.D = np.zeros((self.n_sample-1,self.n_sample))
            for i in range(self.n_sample-1):
                self.D[i,i] = 1
                self.D[i,i+1] = 1
            return self._QuadracticVariationReduction()
        else:
            raise ValueError ('method not supported')
           
    def _median_filter(self,sig):
        r = self.r
        l = len(sig)
        baseline = [] 
        for i in range(l):
            if i < r:
                interval = np.zeros(2*r+1)
                interval[r-i:] = sig[:r+i+1]
                m = np.median(interval)
                baseline.append(m)
            elif i + r > l:
                interval = np.zeros(2*r+1)
                interval[:l-i+r] = sig[i-r:]
                baseline.append(m)
            else:
                interval = sig[i-r:i+r+1]
                baseline.append(m)
        baseline = np.array(baseline)
        return (baseline,sig-baseline)

    def _QuadracticVariationReduction(self):
        baseline_matrix = inv(np.eye(self.n_sample)+self.lamda*np.matmul(self.D.T,self.D))
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
    def __init__(self,data,metadata,T):
        
        '''
        Initialize outliner detection instance
        
        Parameters:
        -----------
        data : ndarray
        raw data or processed data
        metadata : dict
        just call the EMG_Signal.metadata
        T : float
        period of epoch 
        
        '''
        
        super().__init__(data,metadata)
        self.segment(t_Epoch=T)
        
    def outliner_detection(self,**kwargs):
        epochs = self.Epoch
        relativeLowFreqPower = []
        relativePLIPower = []
        rms = []
        for epoch in epochs:
            freq_Epoch,PSD_Epoch = FeatureExtraction.F_Domain(epoch,self.metadata).PSD()
            relativeLowFreqPower.append(self._relativeLowFreqComponentPower(freq_Epoch,PSD_Epoch,**kwargs))
            relativePLIPower.append(self._relativePLIPower(freq_Epoch,PSD_Epoch,**kwargs))
            rms.append(FeatureExtraction.T_Domain(epoch,self.metadata).RMS())
    
        self.rLFP = self._average_feature(relativeLowFreqPower)
        self.rPLIP = self._average_feature(relativePLIPower)
        self.RMS = self._average_feature(rms)
    
    def reference_selection(self):
        LFP_ref = self._reference_criterion(self.rLFP)
        PLI_ref = self._reference_criterion(self.rPLIP)
        self.ref_ch = list(set(LFP_ref) & set(PLI_ref))
    
    def threshold_calculation(self,k1,k2,k3):
        Q_LFP = self.Interquartile_Range([self.rLFP[i] for i in self.ref_ch])
        Q_PLI = self.Interquartile_Range([self.rPLIP[i] for i in self.ref_ch])
        thres_LFP = k1*(Q_LFP[1]+1.5*(Q_LFP[2]-Q_LFP[0]))
        thres_PLI = k2*(Q_PLI[1]+1.5*(Q_PLI[2]-Q_PLI[0]))
        thres_RMS = self._rmsThreshold(k3)
        self.threshold = (thres_LFP,thres_PLI,thres_RMS)
    
    def detect(self):
        active_channels = list(range(self.n_channel))
        return self._detect(active_channels)
    
    def _detect(self,array):
        bad_ch = []
        bad_ch_rLFP = []
        bad_ch_rPLIP = []
        bad_ch_RMS = []
        good_ch = []
        self.ch_info = {}
        a = 0
        for ch in range(len(array)):
            if self.rLFP[ch] > self.threshold[0]:
                a = 1
                bad_ch_rLFP.append(ch)
                if ch not in bad_ch:
                    bad_ch.append(ch)
            if self.rPLIP[ch] > self.threshold[1]:
                a = 1
                bad_ch_rPLIP.append(ch)
                if ch not in bad_ch:
                    bad_ch.append(ch)
            if self.RMS[ch] > self.threshold[2][ch]:
                a = 1
                bad_ch_RMS.append(ch)
                if ch not in bad_ch:
                    bad_ch.append(ch)
            if not a:
                good_ch.append(ch)
        self.ch_info['good_ch'] = good_ch
        self.ch_info['bad_ch'] = bad_ch
        self.ch_info['rLFP'] = bad_ch_rLFP
        self.ch_info['rPLIP'] = bad_ch_rPLIP
        self.ch_info['rms'] = bad_ch_RMS
        return self.ch_info
    
    def _reference_criterion(self,feature):
        ref_ch = []
        Q = self.Interquartile_Range(feature)
        feature_median = Q[1]
        ir = Q[2] - Q[0]
        for ch in range(self.n_channel):
            ch_feature = feature[ch]
            if np.abs(ch_feature-feature_median)<1.5*ir:
                ref_ch.append(ch)
        return ref_ch
    
    def Interquartile_Range(self,data):
        sorted_data = np.sort(data)
        Q2 = np.median(sorted_data)
        left_pos = np.where(sorted_data<Q2)[0][-1]
        right_pos = np.where(sorted_data>Q2)[0][0]
        left = sorted_data[:left_pos+1] 
        right = sorted_data[right_pos:]
        Q1 = np.median(left)
        Q3 = np.median(right)
        return (Q1,Q2,Q3)

    def _average_feature(self,data):    
        for index,i in enumerate(data):
            if index == 0:
                sum = i
            else:
                sum = sum + i
        return sum/len(data)
        
    def _relativeLowFreqComponentPower(self,freq,Power,max_threshold_freq=500,low_threshold_freq=12):
        n_ch = self.n_channel
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
        n_ch = self.n_channel
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
                power_PLI = power_PLI + p[index]
            relativePLIPower.append(power_PLI/power_full)
        return np.array(relativePLIPower)
    
    def _rmsThreshold(self,k):
        thres_RMS = []
        for ch in range(self.n_channel):
            neighbour = self.neighbour[ch]
            mu_EW = self.RMS[neighbour['E_W']].mean()
            std_EW = self.RMS[neighbour['E_W']].std()
            mu_NS = self.RMS[neighbour['N_S']].mean()
            std_NS = self.RMS[neighbour['N_S']].std()
            mu_Other = self.RMS[neighbour['Other']].mean()
            std_Other = self.RMS[neighbour['Other']].std()
            thres_RMS.append(min(mu_EW,mu_NS,mu_Other)+k*max(std_EW,std_NS,std_Other))
        return np.array(thres_RMS)

    def OutlinerVisulize(self,**kwargs):
        row = self.ArrayInfo[0]
        col = self.ArrayInfo[1]
        try:
            radius = kwargs['r']
        except:
            radius = 0.2
        try:
            distance = kwargs['d']
        except:
            distance = (1,0.6)        
        row_d = distance[0]
        col_d = distance[1]
        y = np.arange(-row*row_d,0,row_d)
        x = np.arange(-col*col_d,0,col_d)
        y = y[::-1]
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()     

        for i1,yi in enumerate(y):
            for i2,xi in enumerate(x):
                self.ax.text(xi-0.15,yi+0.3,f'{self.EMap[i1,i2]-1}')
                electrode = self.EMap[i1,i2]-1
                a = electrode in self.ch_info['rLFP']
                b = electrode in self.ch_info['rPLIP']
                c = electrode in self.ch_info['rms']
                colors = ['r','blue','y']
                c1,c2,c3 = colors
                if a and b and c:
                    self.addWedge('bad_tripple',(xi,yi),radius,c1,c2,c3)
                elif (a and b) or (a and c) or (b and c):
                    if not a:
                        self.addWedge('bad_duo',(xi,yi),radius,c2,c3)
                    elif not b:
                        self.addWedge('bad_duo',(xi,yi),radius,c1,c3)
                    else:
                        self.addWedge('bad_duo',(xi,yi),radius,c1,c2)
                elif a or b or c:
                    if a:
                        self.addWedge('bad_solo',(xi,yi),radius,c1)
                    elif b:
                        self.addWedge('bad_solo',(xi,yi),radius,c2)
                    else:
                        self.addWedge('bad_solo',(xi,yi),radius,c3)
                else:
                    self.addWedge('good',(xi,yi),radius)
                    
        red_patch = mpatches.Patch(color='r', label='Low Power')
        blue_patch = mpatches.Patch(color='blue', label='PLI')
        yellow_patch = mpatches.Patch(color='y', label='RMS')
        plt.legend(handles=[red_patch, blue_patch,yellow_patch])
        plt.axis('off')
        self.ax.axis('equal')   
                
    def addWedge(self,type,center,r,*arg):
        if type == 'bad_solo':
            w1 = Wedge(center,r,0,360,fc=arg[0])
            self.ax.add_patch(w1)
        elif type == 'good':
            w1 = Wedge(center,r,0,360,fill=False)
            self.ax.add_patch(w1)
        elif type == 'bad_duo':
            w1 = Wedge(center,r,90,270,fc=arg[0])
            w2 = Wedge(center,r,-90,90,fc=arg[1])
            self.ax.add_patch(w1)
            self.ax.add_patch(w2)
        elif type == 'bad_tripple':
            w1 = Wedge(center,r,-30,90,fc=arg[0])
            w2 = Wedge(center,r,90,210,fc=arg[1])
            w3 = Wedge(center,r,210,-30,fc=arg[2])
            self.ax.add_patch(w1)
            self.ax.add_patch(w2)
            self.ax.add_patch(w3)
        
        
        


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
