import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import sys



class draggable_lines:
    
    '''
    
    Create a draggable line instance on the matplotlib ax
    
    
    Parameters:
    -----------
    ax : matplotlib.axes._axes.Axes
        the axes to create the instance
    type : kwarg, 'h' or 'v'
        whether to create a vertical or horizontal line
    XorY: kwarg, float
        the x or y values for the instance
    ls:  linestyle of matplotlib, default as solid
        linstyle of the instance
    color: color code, default as blue
    
     
    '''
    def __init__(self, ax, ls = 'solid', color = 'blue', **kwarg):
        self.ax = ax
        self.canvas = ax.get_figure().canvas
        
        for kw, v in kwarg.items():
            if kw == 'type':
                self.type = v
            if kw == 'XorY':
                self.XorY = v
        self.linestyle = color
        self.color = ls
        
        if self.type == 'v':
            self.x = [self.XorY,self.XorY]
            lower, upper = ax.get_ybound() 
            self.y = [lower,upper]
        
        elif self.type == 'h':
            self.y = [self.XorY,self.XorY]
            lower, upper = ax.get_xbound() 
            self.x = [lower,upper]
            
        self.line = lines.Line2D(self.x, self.y, picker=5, color = self.color, linestyle= self.linestyle)
        self.ax.add_line(self.line)
        self.canvas.draw()
        self.sid = self.canvas.mpl_connect('pick_event', self.clickonline)

    def clickonline(self, event):
        if event.artist == self.line:
            self.follower = self.canvas.mpl_connect("motion_notify_event", self.followmouse)
            self.releaser = self.canvas.mpl_connect("button_release_event", self.releaseonclick)

    def followmouse(self, event):
        if self.type == "h":
            self.line.set_ydata([event.ydata, event.ydata])
        elif self.type == 'v':
            self.line.set_xdata([event.xdata, event.xdata])       
        self.canvas.draw()

    def releaseonclick(self,event):
        self.canvas.mpl_disconnect(self.releaser)
        self.canvas.mpl_disconnect(self.follower)
        if self.type == "h":
            self.y = self.line.get_ydata()
        elif self.type == 'v':
            self.x = self.line.get_xdata()


class MultiplePlot:
    
    '''
    
    Plot multiple axes given data matrix 
    
    
    Parameters:
    -----------
    nrow : int
        number of rows 
    ncol : int
        number of columns
    ws : float
        horizontal distance between axes, default to 0.8
    hs : float
        vertical distance between axes, default to 1 
    xdata : ndarray
        1D array means it would be applied on all axes, otherwise 2D array with nrow*ncol rows
    ydata : ndarray
        array containing infprmation of y for each ax
    'exist' : {0 or 1}
        plot when all the relevant data is stored already. Internal callable.
    'single' : {0 or 1}
        plot single ax if all the relevant data is stored. Internal callable.
        
    '''
    
    
    def __init__(self,nrow,ncol,ws=0.8,hs=1):
        fig, axes = plt.subplots(nrows=nrow,ncols=ncol)
        self.fig = fig
        self.axes = axes
        self.ncol = ncol
        self.nrow = nrow
        self.double_click = None
        self.double_click_index = None
        plt.subplots_adjust(wspace=ws, hspace=hs)
        
    def main_plot(self,*arg,**kwargs):
        if 'exist' in kwargs:
            if not hasattr(self,'xdata'):
                raise KeyError ('Invalid kwarg')
            elif 'single' in kwargs:
                if len(self.axes) != 1:
                    raise TypeError ('more than 1 ax')
                else:
                    return self._single_main_plot()
            return self._further_main_plot()
        elif len(arg) != 5:
            raise TypeError ('nunber of input should be 5')
        else:
            return self._initial_main_plot(*arg)
    
    def _single_main_plot(self):
        y = self.ydata[self.double_click_index,:]
        x = self.xdata if self.share_x else self.xdata[self.double_click_index]
        xtitle = self.xlabel if self.share_xlabel else self.xlabel[self.double_click_index]
        ytitle = self.ylabel if self.share_ylabel else self.ylabel[self.double_click_index]
        title_ax = self.title[self.double_click_index]
        self.axes[0].plot(x,y)
        self.axes[0].set_xlabel(xtitle)
        self.axes[0].set_ylabel(ytitle)
        self.axes[0].set_title(title_ax)
        
        
    def _further_main_plot(self):
        for num, ax in enumerate(self.axes.flat):
            sig_y = self.ydata[num]
            sig_x = self.xdata if self.share_x else self.xdata[num] 
            if sig_y.ndim  == 1:
                ax.plot(sig_x,sig_y)                
            elif sig_y.ndim == 2:
                for i in range(sig_y.shape[0]):
                    ax.plot(sig_x,sig_y[i,:])                
            else:
                raise ValueError ('dimensions of ydata greater as 2')
            xtitle = self.xlabel if self.share_xlabel else self.xlabel[num]
            ytitle = self.ylabel if self.share_ylabel else self.ylabel[num]
            title_ax = self.title[num]
            ax.set_title(title_ax)
            ax.set_xlabel(xtitle)
            ax.set_ylabel(ytitle)
        
    def _initial_main_plot(self,*arg):
        self.share_x = False
        self.share_xlabel = False
        self.share_ylabel = False
        
        self.xdata = arg[0]
        self.ydata = arg[1]
        self.xlabel = arg[2]
        self.ylabel = arg[3]
        self.title = arg[4]        
        
        try:
            self.ydata.shape[1]
            if self.ydata.shape[0] != self.nrow*self.ncol:
                raise ValueError ('length of data is not equal to number of subplots')
        except:
            if self.nrow*self.ncol != 1:
                raise ValueError ('length of data is not equal to number of subplots')
        
        if self.xdata.ndim > 2:
            raise ValueError ('dimensions of xdata greater as 2')
        elif self.xdata.ndim == 1:
            self.share_x = True

        
        if isinstance(self.xlabel,str):
            self.share_xlabel = True
        else:
            if len(self.xlabel) != self.nrow*self.ncol:
                raise ValueError ('length of xlabel is wrong')
 
        if isinstance(self.ylabel,str):
            self.share_ylabel = True
        else:
            if len(self.ylabel) != self.nrow*self.ncol:
                raise ValueError ('length of xlabel is wrong')       
        
        for num, ax in enumerate(self.axes.flat):
            sig_y = self.ydata[num]
            sig_x = self.xdata if self.share_x else self.xdata[num] 
            if sig_y.ndim  == 1:
                ax.plot(sig_x,sig_y)                
            elif sig_y.ndim == 2:
                for i in range(sig_y.shape[0]):
                    ax.plot(sig_x,sig_y[i,:])                
            else:
                raise ValueError ('dimensions of ydata greater as 2')
            xtitle = self.xlabel if self.share_xlabel else self.xlabel[num]
            ytitle = self.ylabel if self.share_ylabel else self.ylabel[num]
            title_ax = self.title[num]
            ax.set_title(title_ax)
            ax.set_xlabel(xtitle)
            ax.set_ylabel(ytitle)
            

class MultipleWithZoomedSubaxPlot(MultiplePlot):
     
    '''
    
     
    Plot multiple axes with zoomed subaxes
    
    
    
    Parameters:
     -----------
    zoom_lim : tuple or list of tuple
        state the zoom limits of each sub ax
    bounds : list
        positon of the sub ax, default to right upper corner
    select : list
        which ax has sub ax, default to 'all'
     
    '''

    
    def zoomed_subax(self,zoom_lim,bounds=[0.6,0.8,0.4,0.2],select = 'all'): 
        try:
            for i in bounds:
                if i >= 1:
                    raise ValueError ('values in bounds must be smaller as 1')
        except:
            raise TypeError ('bounds is not iterable')
        
        if select == 'all':
            self.select = np.array(range(self.ncol*self.nrow)) 
        elif select[-1] > self.ncol*self.nrow:
            raise ValueError ('select may greater than dimension')
        else:
            self.select = select
        
        if isinstance(zoom_lim,tuple):
            if len(zoom_lim) != 2:
                raise ValueError ('length of zoom_lim should be 2')
            else:
                self.zoom_share_x = True
        else:
            if len(zoom_lim) != len(select):
                raise ValueError ('length of select and zoom_lim not equal')
        
        for num, ax in enumerate(self.axes.flat):
            count = 0
            if num in self.select:
                axin = ax.inset_axes(bounds)
                sig_y = self.ydata[num]
                sig_x = self.xdata if self.share_x else self.xdata[num] 
                lim = zoom_lim if self.zoom_share_x else zoom_lim[count]
                axin.set_xlim(lim[0],lim[1])
                
                if sig_y.ndim  == 1:
                    axin.plot(sig_x,sig_y)                
                elif sig_y.ndim == 2:
                    for i in range(sig_y.shape[0]):
                        axin.plot(sig_x,sig_y[i,:])                
                else:
                    raise ValueError ('dimensions of ydata greater as 2')
                            
                count += 1


class MultipleClickablePlot(MultiplePlot):
    
    
    '''
    
    
    Plot clickable multiple axes
    
    
    Usage :
    create MultipleClickablePlot instance same as MultiplePlot
    call connect_event() method
    

    '''
    
    
    def connect_event(self):
        return self._connect_on_click_event()
    
    def _connect_on_click_event(self):
        self.cid_zoom = self.fig.canvas.mpl_connect('button_press_event',lambda event: self._on_click(event))
    
    def _connect_back_click_event(self):
        self.cid_back = self.fig.canvas.mpl_connect('button_press_event',lambda event: self._back_click(event))
        
    def _on_click(self,event):
        if event.dblclick:
            for index,i in enumerate(self.axes.flat):
                if i == event.inaxes:
                    self.double_click = 1
                    self.double_click_index = index  
                    self.fig.clear()  
                    self.refresh()
                    self.fig.add_subplot(111)
                    self.axes = np.array(self.fig.axes)
                    self.main_plot(exist=1,single=1)
            self.fig.canvas.mpl_disconnect(self.cid_zoom)
            self._connect_back_click_event()        
    
    def _back_click(self,event):
        if event.dblclick:
            self.double_click = None
            self.double_click_index = None
            self.fig.clear()  
            self.refresh()
            for i in range(self.ncol*self.nrow):
                self.fig.add_subplot(self.nrow,self.ncol,i+1)
            self.axes = np.array(self.fig.axes)
            self.main_plot(exist=1)
            self.fig.canvas.mpl_disconnect(self.cid_back)
            self._connect_on_click_event()   
                    
    def _refresh(self):
        plt.pause(0.1)
        plt.draw()


class MultiChannelPlot():
    
    def __init__(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        self.fig = fig
        self.ax = ax
    
    def plot(self,freq,data,ch_name,xtitle='time',ytitle='Channal name',**kwarg):
        if data.shape[0] != len(ch_name):
            raise ValueError ('length of channel names must equal to number of channels')
        
        self.nch = len(ch_name)
        self.data = data
        self.ch_name = ch_name
        self.xtitle = xtitle
        self.ytitle = ytitle
        self.dy = (self.data.min()-self.data.max())
        if 'title' in kwarg:
            self.title = kwarg['title']
        if 'range' in kwarg:
            self.range = kwarg['range']
        self.bad_ch_list = kwarg['bad_channel_list'] if 'bad_channel_list' in kwarg else []
        T = 1/freq
        self.x = np.arange(0,data.shape[1])*T 
        return self._plot()
        
    def _plot(self):
        yticks = []
        for i in range(self.nch):
            yticks.append(i*self.dy)
        self.ax.set_yticks(yticks, labels=self.ch_name)

        for j in range(self.nch):
            c = '#C82423' if j in self.bad_ch_list else '#2878B5'
            self.ax.plot(self.x,self.data[j,:]+j*self.dy,color=c)
        self.ax.set_xlabel(self.xtitle)
        self.ax.set_ylabel(self.ytitle)
        if hasattr(self,'title'):
            self.ax.set_title(self.title)
        if hasattr(self,'range'):
            self.ax.set_xlim(self.range[0],self.range[1])


