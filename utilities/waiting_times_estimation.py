#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:34:04 2023

@author: mdouaihy
"""

import numpy as np

def  waiting_times_estimation(i, time, TT, T0, tmax, PosPred, fParam, w):
    
    content = np.load(fParam)
    FreqEchSimu = content['FreqEchSimu']
    TailleSeqMarq = content['TailleSeqMarq']
    TaillePostMarq = content['TaillePostMarq']
    Polym_speed = content['Polym_speed']


    """ the signal support is [T0(i),tmax(i)]    """
    times2 = time[0] + (np.where(PosPred[:,i] == 1)[0] / FreqEchSimu)/60 
    times1 = times2 -(TailleSeqMarq+TaillePostMarq)/Polym_speed/60 
    
    """ polymerases contribute to times in the interval [times1, times2] 
         times1 < times2 always
         compute waiting times in moving window """
    mw = np.zeros( TT.shape)# will contain the sum of waiting times vs time from one nucleous
    sum_wt_2 = np.zeros( TT.shape)# will contain the sum of waiting times squared vs time from one nucleous
    nintervals = np.zeros(TT.shape)# will contain numbers of gaps from one nucleous
    for j in range(len(TT)):
        l =  max([min(TT),T0[i],TT[j]-w]) # lower bound
        L =  min([max(TT),tmax[i], TT[j]+w]) # upper bound
        """ intersection of window with the signal support is the interval [l,L] """
        if L > l: # if intersection nonempty
            ind= np.where((times2 > l) & (times1 < L))[0] # these positions contribute to signal in window
            tsort = np.sort(np.concatenate((np.array([l,L]),times2[ind])))
            gaps = np.diff(tsort) # waiting times
            mw[j]= np.sum(gaps) # sum waiting times
            sum_wt_2[j] = np.sum(gaps**2) # sum of squared waiting time
            nintervals[j] = len(gaps)# 	
        else:
            mw[j] = np.nan # if intersection empty
            nintervals[j] = 0 

    return [mw, sum_wt_2, nintervals]
