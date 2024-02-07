from siganl_processing_toolbox import *
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
#from scipy import signal
from sklearn.decomposition import PCA
import time
import random

def f():
    print('hh')
    
t = FIR_Filter(f)
final = t(0.3,0.2,50,type='highpass')
w1 = t.WindowFunc
h1 = t.IdealFunc
print(t.order)
#FIR_Filter.BodePlot(final)


   

FIR_Filter.BodePlot(final)
   
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.plot(w1,label='Hanning')
# ax.plot(w2,label='Hanning test')
# ax.plot(h1,label='ideal')
# ax.plot(h2,label='ideal test')
# ax.plot(final,label='last')
# ax.plot(final2,label='last test')
# ax.legend()
plt.show()





'''
x1,y1 = sinusoid(1,0,6*np.pi,2000)
x2,y2 = sinusoid(1,0,6*np.pi,2000,A=2,phi=1)
x3,y3 = sinusoid(2,0,6*np.pi,2000,phi=1)


l1 = 2*y1 + y2 + y3 + np.random.normal(0,2,2000)
l2 = 3*y2 + 2*y1 + 2*y3+ np.random.normal(0,2,2000)
l3 = y1 + y2 - y3 + np.random.normal(0,2,2000)
                  
X = data_concate(l1,l2,l3)

d = EMG_Signal(X,f=10)
f,P = d.PSD(X)


fig = plt.figure()
ax = fig.add_subplot()
for i in range(3):
    ax.plot(f[i,:],P[i,:],label=f'{i}')

ax.legend()
plt.show()

'''


'''
### spike noise test
x,baseline = sinusoid(0.1,0,6*np.pi,2000,A=0.5)
sig = signal(baseline,f=0.1)
s_noise = sig.noise_generator('spikes',multiple=4,height=0.1,width=0.1,range=[20,8000],position='random')
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(x,s_noise+baseline)
plt.show()


'''


'''
x,baseline = sinusoid(0.1,0,6*np.pi,2000,A=0.1,offset=0.2)

x,raw = sinusoid(10,0,6*np.pi,2000,A=0.2)

sig = raw + baseline

data = signal(sig)
b,data = data.detrend('qvr',lamda=0.1)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,sig,label='raw data')
ax.plot(x,data,label='denoise')
ax.plot(x,b,label='baseline')
ax.legend()

plt.show()
'''


'''

## CCA test

start_time = time.time()
x1,y1 = sinusoid(0.5,0,6*np.pi,2000)
x2,y2 = sinusoid(1,0,6*np.pi,2000,A=2,phi=1)
x3,y3 = sinusoid(2,0,6*np.pi,2000,phi=1)
#noise = np.random.normal(0,0.08,2000)

l1 = 2*y1 + y2 + y3 + np.random.normal(0,2,2000)
l2 = 3*y2 + 2*y1 + 2*y3+ np.random.normal(0,2,2000)
l3 = y1 + y2 - y3 + np.random.normal(0,2,2000)
#l4 = -y1 + 2*y2 - y3 + np.random.normal(0,1,2000)

l4 = 2*y1 + y2 + y3 + np.random.normal(0,2,2000)
l5 = 3*y2 + 2*y1 + 2*y3+ np.random.normal(0,2,2000)
l6 = y1 + y2 - y3 + np.random.normal(0,2,2000)

X = data_concate(l1,l2,l3)
Y = data_concate(l4,l5,l6)


algo = CCA(X,Y,10)
algo.fit()
output = algo.decompose()


ch_name = ['ch1','ch2','ch3']

#algo.canonical_component_visulize(output,ch_name,title='canonical component')



output = algo.reconstruct(output,0.5)
end_time = time.time()

print('start: {}'.format(start_time))
print('end: {}'.format(end_time))
print(end_time-start_time)

algo.canonical_component_visulize(output,ch_name,title='denoised')

algo.canonical_component_visulize(X,ch_name,title='raw')


fig = plt.figure()
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(323)
ax3 = fig.add_subplot(325)
#ax4 = fig.add_subplot(427)

ax1.plot(x1,l1)
ax2.plot(x2,l2)
ax3.plot(x1,l3)
#ax4.plot(x1,l4)

ax5 = fig.add_subplot(322)
ax6 = fig.add_subplot(324)
ax7 = fig.add_subplot(326)
#ax8 = fig.add_subplot(428)
#ax4.plot(x1,output[0,:])
#ax5.plot(x1,output[1,:])
#ax6.plot(x1,output[2,:])

ax5.plot(x1,output[0,:])
ax6.plot(x1,output[1,:])
ax7.plot(x1,output[2,:])
#ax8.plot(x1,output[3,:])
plt.show()



'''



'''
## PCA test

x1,y1 = sinusoid(0.5,0,6*np.pi,2000)
x2,y2 = sinusoid(1,0,6*np.pi,2000,A=2,phi=1)
x3,y3 = sinusoid(2,0,6*np.pi,2000,phi=1)
#noise = np.random.normal(0,0.08,2000)

l1 = 2*y1 + y2 + y3 + np.random.normal(0,1,2000)
l2 = 3*y2 + 2*y1 + 2*y3+ np.random.normal(0,1,2000)
l3 = y1 + y2 - y3 + np.random.normal(0,1,2000)
l4 = -y1 + 2*y2 - y3 + np.random.normal(0,1,2000)

data = data_concate(l1,l2,l3,l4)

pca = PCA2(data)
pca.fit(method='SVD')
output = pca.compute(n_component=2)
output = pca.reconstruct()

fig = plt.figure()
ax1 = fig.add_subplot(421)
ax2 = fig.add_subplot(423)
ax3 = fig.add_subplot(425)
ax4 = fig.add_subplot(427)

ax1.plot(x1,l1)
ax2.plot(x2,l2)
ax3.plot(x1,l3)
ax4.plot(x1,l4)

ax5 = fig.add_subplot(422)
ax6 = fig.add_subplot(424)
ax7 = fig.add_subplot(426)
ax8 = fig.add_subplot(428)
#ax4.plot(x1,output[0,:])
#ax5.plot(x1,output[1,:])
#ax6.plot(x1,output[2,:])

ax5.plot(x1,output[0,:])
ax6.plot(x1,output[1,:])
ax7.plot(x1,output[2,:])
ax8.plot(x1,output[3,:])
plt.show()

'''






'''
## Cross- and Auto Correlation test

x1,y1 = sinusoid(0.5,0,6*np.pi,2000)
#x2,y2 = sinusoid(2,0,6*np.pi,2000)
#x3,y3 = sinusoid(3,0,6*np.pi,2000)
x2,y2 = sinusoid(0.5,0,6*np.pi,2000,phi=np.pi)

noise = np.random.rand(y1.shape[0])/2
a = y1 + noise
b = y2 + noise


fig = plt.figure()
#corr = signal.correlate(a, b,method='direct')
c = correlation(a,b,center=True)
ax1 = fig.add_subplot(211)
ax1.plot(c,label='self')
#ax1.plot(corr,label='np')
ax1.legend()
ax2 = fig.add_subplot(212)
ax2.plot(x1,a)
ax2.plot(x2,b)

plt.show()


'''
