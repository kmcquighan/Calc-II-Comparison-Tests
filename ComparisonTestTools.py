# -*- coding: utf-8 -*-
"""
Kelly McQuighan 2017.

These tools can be used to visualize the concept of the comparison test.
"""

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from numpy import *
import matplotlib as mpl
mpl.rcParams['font.size'] = 20
colors = ['#D06728', '#D9A621', '#008040',  '#0080FF', '#A90000', '#7B00F1']

# helper function to plot the arrow head correctly
def plot_arrow(ani,sni,sn_iMinus1,i,n,ax1,ax2,offset,color):
    hl = min([0.1, 0.6*abs(ani)])
    #hw = max(hl,0.05)
    ax1.arrow(i+1+offset, 0, 0, ani-np.sign(ani)*hl, head_width=0.1, head_length=abs(hl),linewidth=5, fc=colors[color], ec=colors[color])
    ax2.arrow(n+offset, sn_iMinus1, 0, ani-np.sign(ani)*hl, head_width=0.1, head_length=abs(hl),linewidth=5, fc=colors[color], ec=colors[color])
    

"""
This function plots the discrete case based on the choice for right endpoint.
It is a helper function that is called by both smallDiscrete and largeDiscrete
where the only difference is how the right endpoint is scaled.
"""   
def plotDiscrete(f,n,nmax,ax1,ax2,ax3,offset,color_start):
    
    func = eval("lambda n: " + f)
    ms = 15
      
    ns = np.linspace(1,nmax,nmax)
    an = func(ns)
    
    sn = np.zeros(nmax)
    sn[0] = an[0]

    plot_arrow(an[0],sn[0],0,0,n,ax1,ax2,offset,color_start+2)
        
    for i in range(1,n):
        sn[i] = sn[i-1]+an[i]

        plot_arrow(an[i],sn[i],sn[i-1],i,n,ax1,ax2,offset,color_start+2)
            
    for i in range(n,nmax):
        sn[i] = sn[i-1]+an[i]
    
    ax1.plot(ns,an,marker='o',linewidth=0,markersize=ms,color=colors[color_start],label=f)
    ax3.plot(np.log(an[n-1]),0.,marker='o',linewidth=0,markersize=ms,color=colors[color_start],label=f)
    
    ax2.plot(ns[:n],sn[:n],marker='o',linewidth=0,markersize=ms,color=colors[color_start],label=f)
    ax2.plot(n,sn[n-1],marker='o',linewidth=0, markersize=ms, color=colors[color_start+2])
    
    return [min(an),max(an),min(sn),max(sn)]


def format_axes(ax1,ax2,ax3, min_an,max_an,min_sn,max_sn,nmax,an,bn):
    ax1.set_xlim([0,1.03*nmax])
    ax2.set_xlim([0,1.03*nmax])
    #ax3.set_xlim([0,1.03*nmax])
    #ax3.set_xlim([np.log(min_an),np.log(max_an)])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.xaxis.set_ticks_position('bottom')
   
    if min_an>0.:
        ax1.set_ylim([0., 1.1*max_an])
    elif max_an<0.:
        ax1.set_ylim([1.1*min_an, 0.])
    else:
        ax1.set_ylim([1.1*min_an, 1.1*max_an])

    if min_sn>0.:
        ax2.set_ylim([0., 1.1*max_sn])
    elif max_sn<0.:
        ax2.set_ylim([1.1*min_sn, 0.])
    else:
        ax2.set_ylim([1.1*min_sn-0.1, 1.1*max_sn+0.1])
    
    #ax3.set_ylim([np.log(0.9*min_an),np.log(1.1*max_an)])
    ax3.set_ylim([-0.1,0.1])
    
    ax1.axhline(0.,color='k',linewidth=1)
    ax2.axhline(0.,color='k',linewidth=1)
    #ax3.axhline(0.,color='k',linewidth=1)
    #ax3.set_yscale("log")
    ax1.set_xlabel('n', fontsize=36)
    ax1.set_title(r'$a_n$', fontsize=36, y=1.1)
    ax2.set_xlabel('k', fontsize=36)
    ax3.set_xlabel(r'$\log a_n$, $\log b_n$', fontsize=36)
    ax2.set_title(r'$s_k=\sum_{n=1}^k a_n$ and $s_k=\sum_{n=1}^k b_n$',fontsize=36, y=1.1)
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.suptitle(r'$a_n$ = '+an+r' $b_n$ = '+bn, fontsize=36, y=1.0)
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20)
    
    ax1.legend(numpoints=1,loc=1)
    ax2.legend(numpoints=1,loc=4)
    
"""
This function compares two series
""" 
def compare(an,bn,n):     
    
    n = int(n)
    maxn = 21
    
    fig = plt.figure(figsize=(20, 7))
    
    ax1 = plt.subplot2grid((10,2), (0, 0),rowspan=7)
    ax2 = plt.subplot2grid((10,2), (0, 1),rowspan=10)
    #ax1 = fig.add_subplot(2,2,1)
    #ax2 = fig.add_subplot(2,2,2,rowspan=2)
    ax3 = plt.subplot2grid((10,2), (7, 0),rowspan=3)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    
    [min_an,max_an,min_sna,max_sna] = plotDiscrete(an,n,maxn,ax1,ax2,ax3,-0.2,0)
    [min_bn,max_bn,min_snb,max_snb] = plotDiscrete(bn,n,maxn,ax1,ax2,ax3,0.2,3)
    
    format_axes(ax1,ax2,ax3,min(min_an,min_bn),max(max_an,max_bn),
                min(min_sna,min_snb),max(max_sna,max_snb),maxn,an,bn)

    