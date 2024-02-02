#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:17:16 2022

@author: mdouaihy
"""

import numpy as np

def sumSignal1_par_gillespie(Trans_positions, fparam, frame_num):
    
    pcontent = np.load(fparam)
    FreqEchSimu = pcontent['FreqEchSimu']
    FreqEchImg = pcontent['FreqEchImg']
    TaillePreMarq = pcontent['TaillePreMarq']
    TailleSeqMarq = pcontent['TailleSeqMarq']
    TaillePostMarq = pcontent['TaillePostMarq']
    Intensity_for_1_Polym = pcontent['Intensity_for_1_Polym']
    Polym_speed = pcontent['Polym_speed']
### compute signal from positions
### compute signal from positions
    Taille = (TaillePreMarq+TailleSeqMarq+TaillePostMarq)
    Sum_signals_matrix = np.zeros((frame_num,len(Trans_positions)))
    ximage = (np.transpose(np.tile(1+np.arange(frame_num), (len(Trans_positions),1))))/FreqEchImg*Polym_speed #### frame positions in bp
    xpos = np.multiply(np.divide((Trans_positions+1),FreqEchSimu),Polym_speed)-Taille
    t1=np.tile(xpos+TaillePreMarq, (frame_num,1))
    ypos = np.subtract(ximage, t1)
    ind = np.logical_and((ypos > 0),(ypos < (TailleSeqMarq + TaillePostMarq))) #

    Sum_signals_matrix[ind] = Sum_signals_matrix[ind] + Signal_par(ypos[ind]-1,Intensity_for_1_Polym,TailleSeqMarq)
    Sum_signals=np.sum(Sum_signals_matrix, axis = 1)
    return Sum_signals

def Signal_par(ypos,Intensity_for_1_Polym,TailleSeqMarq):
    S = np.ones(len(ypos))*Intensity_for_1_Polym
    ind2 = np.where(ypos < TailleSeqMarq)[0]
    S[ind2] = (1+ypos[ind2])/TailleSeqMarq*Intensity_for_1_Polym      
    return S
