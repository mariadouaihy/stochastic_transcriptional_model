#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:53:05 2024

@author: mdouaihy
"""


import numpy as np
import os
import pandas as pd
from movies_combining import movies_combining
from waiting_times_estimation import waiting_times_estimation
from scipy.stats import norm
from scipy.interpolate import interp1d
#plt.close('all')


def ponxkini_perMovie(xlsFile, resultPath_i, proteinfile, fParam, w, TT_int, nc='14', tfinal = 10000):
    content = np.load(resultPath_i)
    DataExp = content['DataExp']
    DataPred = content['DataPred']
    PosPred = content['PosPred']
    
    n = DataExp.shape

    ### loading time into mitosis    
    rawData = pd.read_excel(xlsFile)
    rawData.columns = [x.lower() for x in rawData.columns]
    time_into_mitosis = rawData['time'].dropna(how='all').to_numpy()
        
    if len(time_into_mitosis) ==0:
        print('!!!!! framelen pre-assigned to 3.86')
        tstart = 0
        tend = min(tfinal*60, n[0]*3.86)
        FrameLen = 3.86
    else:
        tstart = time_into_mitosis[0]
        tend = min(tfinal*60, time_into_mitosis[-1])
        FrameLen = np.unique(np.round(np.diff(time_into_mitosis),3))[0]
            
            
    content = np.load(fParam)
    FreqEchImg = content['FreqEchImg']
    TaillePreMarq = content['TaillePreMarq']
    TailleSeqMarq = content['TailleSeqMarq']
    TaillePostMarq = content['TaillePostMarq']
    Polym_speed = content['Polym_speed']
    FreqEchImg = content['FreqEchImg']
    FreqEchSimu = content['FreqEchSimu']
    EspaceInterPolyMin = content['EspaceInterPolyMin']
    DureeSignal = content['DureeSignal']
    frame_num = round(tend/FrameLen)-1    
    
    
    DureeSimu = frame_num*FrameLen ### film duration in s
    DureeAnalysee = DureeSimu + DureeSignal

    num_possible_poly = round(DureeAnalysee/(EspaceInterPolyMin/Polym_speed))
    
    dataExp = np.zeros((frame_num+1, n[1]))
    dataPred = np.zeros((frame_num+1, n[1]))
    posPred= np.zeros((num_possible_poly, n[1]))
    
    n3 = PosPred.shape
    
    if tstart!=0:
        dataExp[int(np.round(tstart/FrameLen)): round(tstart/FrameLen)+frame_num, :] = DataExp[:min((frame_num-round(tstart/FrameLen))+1, DataExp.shape[0]),:]
        dataPred[int(np.round(tstart/FrameLen)): round(tstart/FrameLen)+frame_num, :] = DataPred[:min((frame_num-round(tstart/FrameLen))+1, DataExp.shape[0]),:]
        t0_posPred = round(tstart*FreqEchSimu)
        
        if (t0_posPred+n3[0]-num_possible_poly) <=0:
            posPred[t0_posPred: t0_posPred+n3[0], :] = PosPred[:num_possible_poly-t0_posPred, :]
        else:
            posPred[t0_posPred: t0_posPred+n3[0], :] = PosPred[:-(t0_posPred+n3[0]-num_possible_poly),:]

    
    DataExp = dataExp.copy()
    DataPred = dataPred.copy()
    PosPred = posPred.copy()
    
    DataExp = DataExp[:min(int(tfinal*60/FrameLen), DataExp.shape[0]),:]
    DataPred = DataPred[:min(int(tfinal*60/FrameLen), DataExp.shape[0]),:]
    PosPred = PosPred[:min(int(tfinal*60*FreqEchSimu), PosPred.shape[0]),:]
    
    n = DataExp.shape
    
    T0_movie=np.zeros((n[1],))  ##### will contain the start of the analyzed region   
    for i in range(n[1]): ### for all cells
        pospol =  np.where(PosPred[:,i] == 1)[0] 
        times = pospol / FreqEchSimu  
        ind = np.where( times -  (TailleSeqMarq+TaillePostMarq)/Polym_speed > tend )[0] #### positions that have no influence
        PosPred[pospol[ind],i] = 0 
        max_intensity = max(DataPred[:,i])
        ihit = np.where(DataPred[:,i] > max_intensity/5 )[0]
        if len(ihit)!=0:
            ihi_min =  min(ihit)
            T0_movie[i] = (ihi_min-1)/FreqEchImg; #### T0_combined    
        else:
            T0_movie[i]= tend
            
            
    T0 = np.zeros(DataExp.shape[1])
    tmax = np.zeros(DataExp.shape[1])
#    print(tend)
    for n_i in range(DataExp.shape[1]):
        ihit=np.where(DataPred[:,n_i]> np.nanmax(DataPred[:,n_i])/5)[0]
        if len(ihit)== 0:
            T0[n_i] = T0_movie[n_i]
            tmax[n_i] = tend
        else:
            T0[n_i] =  (min(ihit)+1)/FreqEchImg
            tmax[n_i] = (max(ihit)+1)/FreqEchImg    
    if nc == 'nc14':
        xtime = np.arange(0,n[0])/FreqEchImg/60
        T0_minute=T0/60
        tmax_minute=tmax/60
    else:
        xtime = np.arange(-(n[0]),0,1)/FreqEchImg/60
        T0_minute = xtime[0] + T0/60
        tmax_minute = xtime[0] + tend/60
    

    """ protein extraction """
    if len(proteinfile)!=0:
        
        protdata = pd.read_excel(proteinfile)
        tprot = protdata.iloc[1:,0].tolist() 
        tprot = np.array(tprot)[np.array(tprot)<min(tprot[-1], tfinal)]
        protdata = protdata.iloc[:len(tprot),:]
    else:
        if nc == 'nc14':
            tprot = tmax_minute
        else:
            tprot = T0_minute
            
    """ resample the protein data with the times TTpol where pol2 initiation events are """
    
    nn = DataExp.shape
    
    if nc == 'nc14':
        tt = min(max(tmax_minute),max(tprot))
        tstep =(tt-min(T0_minute))/200
        TT = np.arange(min(T0_minute),tt+tstep,tstep) 
    else:
        tt = max(min(T0_minute), min(tprot))
        if tt == 0:
            tt = max(min(T0_minute), min(T0_minute))
        tstep = tt/200
        TT =  np.arange(tt,-tstep,-tstep) # signal maximal support
#        print(TT)

    TTpol = TT -(TaillePreMarq+TailleSeqMarq+TaillePostMarq)/Polym_speed/60 # where pol2 initiation events are
    
    sum_wt_over_all_cells = np.zeros(TT.shape) # will contain mean 
    sum_wt_2_overl_all_cells = np.zeros(TT.shape) # will contain mean 
    num_gaps = np.zeros(TT.shape) # will contain numbers of waiting times contirbutiing to window mean
    
  
    if len(TT_int) == 0:
        TT_int = TT
    
    nintervals_nuclei = np.empty((nn[1], len(TT_int)))
    nintervals_nuclei[:] = np.nan
    
    sum_wt_per_nuclei = np.empty((nn[1], len(TT_int)))
    sum_wt_per_nuclei[:] = np.nan
    
    sum_wt_2_per_nuclei = np.empty((nn[1], len(TT_int)))
    sum_wt_2_per_nuclei[:] = np.nan 
    
    ponxkini_per_nuclei = np.empty((nn[1], len(TT_int)))
    ponxkini_per_nuclei[:] = np.nan
    
    
    for i in range(nn[1]): # for all cells  in nc14

        [mw, sum_wt_2, nintervals] = waiting_times_estimation(i, xtime, TT_int, T0_minute, tmax_minute, PosPred[round(T0[i]*FreqEchSimu):round(tmax[i]*FreqEchSimu),:], fParam, w)
#        mw = interp1d(interp1d(TTpol_i, mw, fill_value='extrapolate')(TT)TTpol_i, sum_pon_est_i, fill_value='extrapolate')(TT)
#        sum_wt_2
        ind = np.where(~np.isnan(mw))
        sum_wt_over_all_cells[ind]  = sum_wt_over_all_cells[ind] + mw[ind] # perform mean over all cells
        sum_wt_2_overl_all_cells[ind] = sum_wt_2_overl_all_cells[ind] + sum_wt_2[ind]
        num_gaps[ind] = num_gaps[ind] + nintervals[ind] # number of gaps contributing to the
        
        sum_wt_per_nuclei[i, ind] = mw[ind]
        sum_wt_2_per_nuclei[i, ind] = sum_wt_2[ind]
        nintervals_nuclei[i, ind] = nintervals[ind]
        ponxkini_per_nuclei[i,ind] =  nintervals[ind]/mw[ind]
    	

    
    return [TT, TTpol, ponxkini_per_nuclei, num_gaps, sum_wt_over_all_cells, sum_wt_2_overl_all_cells, sum_wt_per_nuclei, sum_wt_2_per_nuclei, nintervals_nuclei]


def ponxkini(w, extension, nc, proteinfile, PathToData, data_type, fParam,bcpd, repressed, filtered = 0, tfinal = 10000):
    
        
    path_to_files = PathToData
    content = np.load(fParam)
    
    FreqEchImg = content['FreqEchImg']
    TaillePreMarq = content['TaillePreMarq']
    TailleSeqMarq = content['TailleSeqMarq']
    TaillePostMarq = content['TaillePostMarq']
    Polym_speed = content['Polym_speed']
    FreqEchImg = content['FreqEchImg']
    FreqEchSimu = content['FreqEchSimu']

    if bcpd == str(0):
        xlspath = path_to_files+ 'rawData/xlsFile/'
        resultpath = path_to_files+ 'resultDec' + data_type +'/'
      
        files=np.array(os.listdir(xlspath))
        files = list(map(lambda x: x.replace(extension,'') ,files))

        [DataExp, DataPred, PosPred, T0_movie, tmax_movie, FrameLen] = movies_combining(xlspath, resultpath, files, fParam, extension)

    
        DataExp = DataExp[:min(int(tfinal*60/FrameLen), DataExp.shape[0]),:]
        DataPred = DataPred[:min(int(tfinal*60/FrameLen), DataExp.shape[0]),:]
        PosPred = PosPred[:min(int(tfinal*60*FreqEchSimu), PosPred.shape[0]),:]
    
        T0 = np.zeros(DataExp.shape[1])
        tmax = np.zeros(DataExp.shape[1])
        for n_i in range(DataExp.shape[1]):
            ihit=np.where(DataPred[:,n_i]> np.nanmax(DataPred[:,n_i])/5)[0]
            if len(ihit)== 0:
                T0[n_i] = T0_movie[n_i]
                tmax[n_i] = tmax_movie[n_i]
            else:
                T0[n_i] =  (min(ihit)+1)/FreqEchImg
                tmax[n_i] = (max(ihit)+1)/FreqEchImg
                
                
    elif repressed ==1:
        
        resultpath = path_to_files +'/result_BCPD_filtered_' + str(filtered) + '.npz'
        pcontent = np.load(resultpath)
        DataExp = pcontent['DataExp_repressed']
        DataPred = pcontent['DataPred_repressed']
        PosPred = pcontent['PosPred_repressed']
        
        T0_b = pcontent['T0_repressed']
        tmax_b = pcontent['tmax_repressed']
        
        T0 = np.zeros(DataExp.shape[1])
        tmax = np.zeros(DataExp.shape[1])
        for n_i in range(DataExp.shape[1]):
            ihit=np.where(DataPred[:,n_i]> np.nanmax(DataPred[:,n_i])/5)[0]
            if len(ihit)== 0:
                T0[n_i] = T0_b[n_i]
                tmax[n_i] = tmax_b[n_i]
            else:
                T0[n_i] = min(T0_b) + (min(ihit)+1)/FreqEchImg
                tmax[n_i] = min(T0_b) + (max(ihit)+1)/FreqEchImg

            
    elif repressed ==0:
        resultpath = path_to_files +'/result_BCPD_filtered_' + str(filtered) + '.npz'
        pcontent = np.load(resultpath)
        DataExp = pcontent['DataExp_activation']
        DataPred = pcontent['DataPred_activation']
        PosPred = pcontent['PosPred_activation']

        tmax_b = []
        for i in range(DataExp.shape[1]):
            tmax_i = np.where(np.isnan(DataExp[:,i]))[0]
            if len(tmax_i)==0:
                tmax_b.append(DataExp.shape[0])
            elif tmax_i[0] ==0:
                tmax_b.append(0)
            else:
                tmax_b.append(tmax_i[0]-1)
        
        T0 = np.zeros(DataExp.shape[1])
        tmax = np.zeros(DataExp.shape[1])
        for n_i in range(DataExp.shape[1]):
            ihit=np.where(DataPred[:,n_i]> np.nanmax(DataPred[:,n_i])/5)[0]
            if len(ihit)== 0:
                tmax[n_i] = tmax_b[n_i]
            else:
                T0[n_i] =  (min(ihit)+1)/FreqEchImg
                tmax[n_i] = (max(ihit)+1)/FreqEchImg

                    

    n = DataExp.shape
    
    if nc == 'nc14':
        xtime = np.arange(0,n[0])/FreqEchImg/60
        T0_minute=T0/60
        tmax_minute=tmax/60
    else:
        xtime = np.arange(-(n[0]),0,1)/FreqEchImg/60
        T0_minute = xtime[0] + T0/60
        tmax_minute = xtime[0] + tmax/60
    

    """ protein extraction """
    if len(proteinfile)!=0:
        
        protdata = pd.read_excel(proteinfile)
        tprot = protdata.iloc[1:,0].tolist() 
        tprot = np.array(tprot)[np.array(tprot)<min(tprot[-1], tfinal)]
        protdata = protdata.iloc[:len(tprot),:]
    else:
        if nc == 'nc14':
            tprot = tmax_minute
        else:
            tprot = T0_minute
            
    """ resample the protein data with the times TTpol where pol2 initiation events are """
    
    nn = DataExp.shape
    
    if nc == 'nc14':
        tt = min(max(tmax_minute),max(tprot))
        tstep =(tt-min(T0_minute))/200
        TT = np.arange(min(T0_minute),tt+tstep,tstep) 
    else:
        tt = max(min(T0_minute), min(tprot))
        if tt == 0:
            tt = max(min(T0_minute), min(T0_minute))
        tstep = tt/200
        TT =  np.arange(tt,-tstep,-tstep) # signal maximal support


    TTpol = TT -(TaillePreMarq+TailleSeqMarq+TaillePostMarq)/Polym_speed/60 # where pol2 initiation events are
    
    sum_wt_over_all_cells = np.zeros(TT.shape) # will contain mean 
    sum_wt_2_overl_all_cells = np.zeros(TT.shape) # will contain mean 
    num_gaps = np.zeros(TT.shape) # will contain numbers of waiting times contirbutiing to window mean
    
  
    
    nintervals_nuclei = np.empty((nn[1], len(TT)))
    nintervals_nuclei[:] = np.nan
    
    sum_wt_per_nuclei = np.empty((nn[1], len(TT)))
    sum_wt_per_nuclei[:] = np.nan
    
    sum_wt_2_per_nuclei = np.empty((nn[1], len(TT)))
    sum_wt_2_per_nuclei[:] = np.nan 
    
    ponxkini_per_nuclei = np.empty((nn[1], len(TT)))
    ponxkini_per_nuclei[:] = np.nan
    
    
    for i in range(nn[1]): # for all cells  in nc14
        [mw, sum_wt_2, nintervals] = waiting_times_estimation(i, xtime, TT, T0_minute, tmax_minute, PosPred[round(T0[i]*FreqEchSimu):round(tmax[i]*FreqEchSimu),:], fParam, w)
        ind = np.where(~np.isnan(mw))
        sum_wt_over_all_cells[ind]  = sum_wt_over_all_cells[ind] + mw[ind] # perform mean over all cells
        sum_wt_2_overl_all_cells[ind] = sum_wt_2_overl_all_cells[ind] + sum_wt_2[ind]
        num_gaps[ind] = num_gaps[ind] + nintervals[ind] # number of gaps contributing to the
        
        sum_wt_per_nuclei[i, ind] = mw[ind]
        sum_wt_2_per_nuclei[i, ind] = sum_wt_2[ind]
        nintervals_nuclei[i, ind] = nintervals[ind]
        ponxkini_per_nuclei[i,ind] =  nintervals[ind]/mw[ind]
    	

    
    return [TT, TTpol, ponxkini_per_nuclei, num_gaps, sum_wt_over_all_cells, sum_wt_2_overl_all_cells, sum_wt_per_nuclei, sum_wt_2_per_nuclei, nintervals_nuclei]



def ponxkini_deconvolveData(tfinal, w, nc, proteinfile, PathToData, data_type, fParam,bcpd, repressed, filtered = '0'):
    
        
    path_to_files = PathToData
    content = np.load(fParam)
    
    FreqEchImg = content['FreqEchImg']
    TaillePreMarq = content['TaillePreMarq']
    TailleSeqMarq = content['TailleSeqMarq']
    TaillePostMarq = content['TaillePostMarq']
    Polym_speed = content['Polym_speed']
    FreqEchImg = content['FreqEchImg']
    FreqEchSimu = content['FreqEchSimu']
    FrameLen = content['FrameLen']
    
    if bcpd == str(0):
        resultpath = path_to_files
        content = np.load(resultpath) 
        DataExp = content['DataExp']
        PosPred = content['PosPred']
        DataPred = content['DataPred']
        
        T0_movie = np.argmax(DataExp != 0, axis=0)*FrameLen 
        tmax_movie = np.ones((DataExp.shape[1],))*FrameLen*DataExp.shape[0]
#        [DataExp, DataPred, PosPred, T0_movie, tmax_movie, FrameLen] = movies_combining(xlspath, resultpath, files, fParam, extension)

        T0 = np.zeros(DataExp.shape[1])
        tmax = np.zeros(DataExp.shape[1])
        for n_i in range(DataExp.shape[1]):
            ihit=np.where(DataPred[:,n_i]> np.nanmax(DataPred[:,n_i])/5)[0]
            if len(ihit)== 0:
                T0[n_i] = T0_movie[n_i]
                tmax[n_i] = tmax_movie[n_i]
            else:
                T0[n_i] =  (min(ihit)+1)/FreqEchImg
                tmax[n_i] = (max(ihit)+1)/FreqEchImg
                
                
    elif repressed ==1:
        
        resultpath = path_to_files +'/result_BCPD_filtered_' + str(filtered) + '.npz'
        pcontent = np.load(resultpath)
        DataExp = pcontent['DataExp_repressed']
        DataPred = pcontent['DataPred_repressed']
        PosPred = pcontent['PosPred_repressed']
        
        T0_b = pcontent['T0_repressed']
        tmax_b = pcontent['tmax_repressed']
        
        T0 = np.zeros(DataExp.shape[1])
        tmax = np.zeros(DataExp.shape[1])
        for n_i in range(DataExp.shape[1]):
            ihit=np.where(DataPred[:,n_i]> np.nanmax(DataPred[:,n_i])/5)[0]
            if len(ihit)== 0:
                T0[n_i] = T0_b[n_i]
                tmax[n_i] = tmax_b[n_i]
            else:
                T0[n_i] = min(T0_b) + (min(ihit)+1)/FreqEchImg
                tmax[n_i] = min(T0_b) + (max(ihit)+1)/FreqEchImg

            
    elif repressed ==0:
        resultpath = path_to_files +'/result_BCPD_filtered_' + str(filtered) + '.npz'
        pcontent = np.load(resultpath)
        DataExp = pcontent['DataExp_activation']
        DataPred = pcontent['DataPred_activation']
        PosPred = pcontent['PosPred_activation']

        tmax_b = []
        for i in range(DataExp.shape[1]):
            tmax_i = np.where(np.isnan(DataExp[:,i]))[0]
            if len(tmax_i)==0:
                tmax_b.append(DataExp.shape[0])
            elif tmax_i[0] ==0:
                tmax_b.append(0)
            else:
                tmax_b.append(tmax_i[0]-1)
        
        T0 = np.zeros(DataExp.shape[1])
        tmax = np.zeros(DataExp.shape[1])
        for n_i in range(DataExp.shape[1]):
            ihit=np.where(DataPred[:,n_i]> np.nanmax(DataPred[:,n_i])/5)[0]
            if len(ihit)== 0:
                tmax[n_i] = tmax_b[n_i]
            else:
                T0[n_i] =  (min(ihit)+1)/FreqEchImg
                tmax[n_i] = (max(ihit)+1)/FreqEchImg

                    

    n = DataExp.shape
    
    if nc == 'nc14':
        xtime = np.arange(0,n[0])/FreqEchImg/60
        T0_minute=T0/60
        tmax_minute=tmax/60
    else:
        xtime = np.arange(-(n[0]),0,1)/FreqEchImg/60
        T0_minute = xtime[0] + T0/60
        tmax_minute = xtime[0] + tmax/60
    

    """ protein extraction """
    if len(proteinfile)!=0:
        
        protdata = pd.read_excel(proteinfile)
        tprot = protdata.iloc[1:,0].tolist() 
        tprot = np.array(tprot)[np.array(tprot)<min(tprot[-1], tfinal)]
        protdata = protdata.iloc[:len(tprot),:]
    else:
        if nc == 'nc14':
            tprot = tmax_minute
        else:
            tprot = T0_minute
            
    """ resample the protein data with the times TTpol where pol2 initiation events are """
    
    nn = DataExp.shape
    
    if nc == 'nc14':
        tt = min(max(tmax_minute),max(tprot))
        tstep =(tt-min(T0_minute))/200
        TT = np.arange(min(T0_minute),tt+tstep,tstep) 
    else:
        tt = max(min(T0_minute), min(tprot))
        if tt == 0:
            tt = max(min(T0_minute), min(T0_minute))
        tstep = tt/200
        TT =  np.arange(tt,-tstep,-tstep) # signal maximal support


    TTpol = TT -(TaillePreMarq+TailleSeqMarq+TaillePostMarq)/Polym_speed/60 # where pol2 initiation events are
    
    sum_wt_over_all_cells = np.zeros(TT.shape) # will contain mean 
    sum_wt_2_overl_all_cells = np.zeros(TT.shape) # will contain mean 
    num_gaps = np.zeros(TT.shape) # will contain numbers of waiting times contirbutiing to window mean
    
  
    
    nintervals_nuclei = np.empty((nn[1], len(TT)))
    nintervals_nuclei[:] = np.nan
    
    sum_wt_per_nuclei = np.empty((nn[1], len(TT)))
    sum_wt_per_nuclei[:] = np.nan
    
    sum_wt_2_per_nuclei = np.empty((nn[1], len(TT)))
    sum_wt_2_per_nuclei[:] = np.nan 
    
    ponxkini_per_nuclei = np.empty((nn[1], len(TT)))
    ponxkini_per_nuclei[:] = np.nan
    
    
    for i in range(nn[1]): # for all cells  in nc14
        [mw, sum_wt_2, nintervals] = waiting_times_estimation(i, xtime, TT, T0_minute, tmax_minute, PosPred[round(T0[i]*FreqEchSimu):round(tmax[i]*FreqEchSimu),:], fParam, w)
        ind = np.where(~np.isnan(mw))
        sum_wt_over_all_cells[ind]  = sum_wt_over_all_cells[ind] + mw[ind] # perform mean over all cells
        sum_wt_2_overl_all_cells[ind] = sum_wt_2_overl_all_cells[ind] + sum_wt_2[ind]
        num_gaps[ind] = num_gaps[ind] + nintervals[ind] # number of gaps contributing to the
        
        sum_wt_per_nuclei[i, ind] = mw[ind]
        sum_wt_2_per_nuclei[i, ind] = sum_wt_2[ind]
        nintervals_nuclei[i, ind] = nintervals[ind]
        ponxkini_per_nuclei[i,ind] =  nintervals[ind]/mw[ind]
    	

    
    return [TT, TTpol, ponxkini_per_nuclei, num_gaps, sum_wt_over_all_cells, sum_wt_2_overl_all_cells, sum_wt_per_nuclei, sum_wt_2_per_nuclei, nintervals_nuclei]


def remove_outliers(data, threshold=1.5):
    # Calculate the first and third quartiles (Q1 and Q3)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Define the lower and upper bounds for outliers
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    # Keep only the data points within the bounds
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]

    return filtered_data

def CI_clt(sum_wt_2_overl_all_cells, sum_pon_est, num_gaps):
    confidence_level = 0.95
    alpha = 1 - confidence_level
    z_critical = norm.ppf(1 - alpha / 2)
    std = np.sqrt((sum_wt_2_overl_all_cells - (sum_pon_est ** 2) / num_gaps) / num_gaps)
    margin_of_error = z_critical * (std / np.sqrt(num_gaps))
    confidence_interval_lower_bd = sum_pon_est/num_gaps - margin_of_error
    confidence_interval_upper_bd = sum_pon_est/num_gaps + margin_of_error
    
    return np.array([confidence_interval_lower_bd, confidence_interval_upper_bd])
