#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:48:49 2023

@author: mdouaihy
"""

import numpy as np
import sys

def smatrix_fct(model):

    if model=='ON_OFF':
        smatrix = np.array([
        [1], #OFF-->ON
        [-1]]) #ON-->OFF

    elif model=='ON_OFF_mRNA_degradation':
        smatrix = np.array([
        [1,0], #OFF-->ON
        [-1,0], #ON-->OFF
        [0,1], #ON--> ON+mRNA
        [0,-1]]) #mRNA -->0
        
 
    elif model=='ON_OFF_mRNA':
        smatrix = np.array([
        [1,0], #OFF-->ON
        [-1,0], #ON-->OFF
        [0,1]]) #ON--> ON+mRNA

    elif model=='ON_OFF_mRNA_protein':
        smatrix = np.array([
        [1,0,0], # OFF--> ON
        [-1,0,0], #ON-->OFF
        [0,1,0], #ON-->ON+mRNA
        [0,-1,0], #mRNA-->0
        [0,0,1], #mRNA-->mRNA+prot
        [0,0,-1]]) #prot-->0
        
    elif model == 'feedback_negative_2S':
        smatrix = np.array([
        [1, 0, 0, 0], # OFF--> ON, 1-s --> s
        [-1,0, 0, 0], # ON-->OFF, s --> 1-s
        [0, 1, 0, 0], # ON-->ON+mRNA
        [0,-1, 0, 0], # mRNA-->0
        [0, 0, 1, 0], # mRNA-->mRNA+prot
        [0, 0,-1, 0], # prot-->0
        [0, 0, 0, 1], # ON --> ON + mRNA_nascent
        [0, 0, 0,-1]  # mRNA_nascent --> mRNA_nascent
        ])
    
    elif model == 'feedback_negative_3S':
        smatrix = np.array([
        [1 ,-1, 0, 0, 0, 0], # OFF1--> ON
        [-1, 1, 0, 0, 0, 0], # ON-->OFF1
        [1 , 0,-1, 0, 0, 0], # OFF2--> ON
        [-1, 0, 1, 0, 0, 0], # ON-->OFF2
        [0 , 0, 0, 1, 0, 0], # ON-->ON+mRNA
        [0 , 0, 0,-1, 0, 0], # mRNA-->0
        [0 , 0, 0, 0, 1, 0], # mRNA-->mRNA+prot
        [0 , 0, 0, 0,-1, 0], # prot-->0
        [0 , 0, 0, 0, 0, 1] # ON --> ON + mRNA_nascent
        ])


    elif model == 'feedback_negative_3S_prot':
        smatrix = np.array([
        [1 ,-1, 0, 0], # OFF1--> ON
        [-1, 1, 0, 0], # ON-->OFF1
        [1 , 0,-1, 0], # OFF2--> ON
        [-1, 0, 1, 0], # ON-->OFF2
        [0 , 0, 0, 1] # ON --> ON + mRNA_nascent
        ])

    
    elif model == 'first_activation_2S':

        smatrix = np.array([
        #inact2, inact1, ON,  OFF, mRNA, prot
         [-1    ,1     , +0 , +0  , +0  , +0], # inact2 --> inact1
         [+1     ,-1   , +0 , +0  , +0  , +0], # inact1 --> inact2
         [+0     ,-1   , +1 , +0  , +0  , +0], # inact1 --> ON
         [+0     ,+0   , -1,  +1  , +0  , +0], # ON     --> OFF
         [+0     ,+0   , +1 , -1  , +0  , +0], # OFF    --> ON
         [+0     ,+0   , +0 , +0  , +1  , +0], # ON     --> ON+mRNA
         [+0     ,+0   , +0 , +0  , -1  , +0], # mRNA   --> 0
         [+0     ,+0   , +0 , +0  , +0  , +1], # mRNA   --> mRNA + prot
         [+0     ,+0   , +0 , +0  , +0  , -1], # prot   --> 0
         ])
    
    else:
        sys.exit("no correct model has been given")
            
    return smatrix




def propensities_feedback_negative_3S_all_hill(pars, state):
    promoter_ON = state[0]
    promoter_OFF1 = state[1]
    promoter_OFF2 = state[2]
    mRNA = state[3]
    protein  = state[4]
    
    

    kr = pars[0]
    lambda_r = pars[1]
    
    kp = pars[2] 
    lambda_p = pars[3]
    
    km1_act = pars[4] 
    km1_rep = pars[5]
    nhkm = pars[6]
    thkm = pars[7]
    k_OFF1 = km1_act + (km1_rep-km1_act)*(protein**nhkm/(thkm**nhkm + protein**nhkm))
    
    kp1_act = pars[8]
    kp1_rep = pars[9] 
    nhkp = pars[10] 
    thkp = pars[11]
    k_ON1 = kp1_act + (kp1_rep-kp1_act)*(protein**nhkp/(thkp**nhkp + protein**nhkp))
    
    km2_act = pars[12] 
    km2_rep = pars[13]
    nhkm2 = pars[14]
    thkm2 = pars[15]
    k_OFF2 = km2_act + (km2_rep-km2_act) *(1 / (1 + (thkm2 / protein) ** nhkm2)) #(protein**nhkm2/(thkm2**nhkm2 + protein**nhkm2))
    
    kp2_act = pars[16]
    kp2_rep = pars[17] 
    nhkp2 = pars[18] 
    thkp2 = pars[19]
    k_ON2 = kp2_act + (kp2_rep-kp2_act) *(1 / (1 + (thkp2 / protein) ** nhkp2)) #*(protein**nhkp2/(thkp2**nhkp2 + protein**nhkp2))
    
    
    

    R=np.zeros((9,))
    
    R[0] = k_ON1 * promoter_OFF1 # OFF1 to ON
    R[1] = k_OFF1 * promoter_ON # ON to OFF1
    
    R[2] = k_ON2* promoter_OFF2 # OFF2 to ON
    R[3] = k_OFF2 * promoter_ON # ON to OFF2
    
    R[4] = kr * promoter_ON # mRNA production
    R[5] = lambda_r * mRNA # mRNA degradation
    
    R[6] = kp * mRNA # protein production
    R[7] = lambda_p * protein # protein degradation
    
    R[8] = kr * promoter_ON # nascent mRNA production
    return R.tolist() 


def propensities_feedback_negative_3S_k2m_prot(pars, state, protein):
    promoter_ON = state[0]
    promoter_OFF1 = state[1]
    promoter_OFF2 = state[2]

    kini = pars[0]    
    kp1_act = pars[1] 
    kp1_rep = pars[2]
    km1 = pars[3]
    kp2_act = pars[4]
    kp2_rep = pars[5]
    km2_act = pars[6] 
    km2_rep = pars[7]
    nhkm2 = pars[8]
    thkm2 = pars[9]
    
    
    k_OFF2 = km2_act + (km2_rep-km2_act) *(1 / (1 + (thkm2 / protein) ** nhkm2)) #(protein**nhkm2/(thkm2**nhkm2 + protein**nhkm2))
    k_ON2 = kp2_act + (kp2_rep-kp2_act) *(1 / (1 + (thkm2 / protein) ** nhkm2)) #(protein**nhkm2/(thkm2**nhkm2 + protein**nhkm2))
    k_ON1 = kp1_act + (kp1_rep-kp1_act) *(1 / (1 + (thkm2 / protein) ** nhkm2)) #(protein**nhkm2/(thkm2**nhkm2 + protein**nhkm2))

    R=np.zeros((5,))
    
    R[0] = k_ON1 * promoter_OFF1 # OFF1 to ON
    R[1] = km1 * promoter_ON # ON to OFF1
    
    R[2] = k_ON2* promoter_OFF2 # OFF2 to ON
    R[3] = k_OFF2 * promoter_ON # ON to OFF2
    
    R[4] = kini * promoter_ON # mRNA production
    
    return R.tolist() 



def gillespie_main(func,state, pars, model, time_points):
    
    smatrix = smatrix_fct(model)
    
    t = 0
    indx_sampling_old = 0
    
    state_update = np.zeros((len(time_points), len(state)))
    finaltime = time_points[-1]
    
    state_immediate_return = state.copy()
    
    while t < finaltime: 
        r1, r2 = np.random.uniform(0,1,2)
        
        a = func(pars, state_immediate_return)
        
        a_cum = np.cumsum(a)
        a_0 = a_cum[-1]
        
        
        tau = (1/a_0)*np.log(1/r1)
        t = t + tau
        
        condition = r2*a_0
        j = np.where(a_cum > condition)[0][0]
        
        state = state_immediate_return + smatrix[j]
        state_immediate_return = state.copy()
        if j == 8:
            state_immediate_return[5] = 0
            state_immediate_return[0] = 1
        
        indx_sampling = np.max(np.where(t>=time_points)[0])
        if indx_sampling< indx_sampling_old:
            continue
        state_update[indx_sampling_old:indx_sampling, :] = np.tile(state,(indx_sampling-indx_sampling_old,1))
        indx_sampling_old = indx_sampling
    state_update = state_update.astype('float16')  
    return state_update 



def gillespie_main_prot_real(func,state, pars, model, time_points, num_possible_poly, tinter, t0 =0, prot=0):
    
    smatrix = smatrix_fct(model)
    
    t = time_points[t0]
    
    indx_sampling_old = t0
    
    state_update = np.zeros((len(time_points), len(state)))
    finaltime = time_points[-1]
    
    state_immediate_return = state.copy()
    polII_pos = np.zeros((num_possible_poly,))  
   
    while t < finaltime: 
            
        r1, r2 = np.random.uniform(0,1,2)
        a = func(pars, state_immediate_return, prot[indx_sampling_old])
        
        a_cum = np.cumsum(a)
        a_0 = a_cum[-1]
        
        
        tau = (1/a_0)*np.log(1/r1)
        t = t + tau
        
        condition = r2*a_0
        j = np.where(a_cum > condition)[0][0]
        
        state = state_immediate_return + smatrix[j]
        state_immediate_return = state.copy()
        if j == 4:
            state_immediate_return[3] = 0
            state_immediate_return[0] = 1
            ipos=int(t/tinter) 
            polII_pos[ipos]=1 
            
        indx_sampling = np.max(np.where(t>=time_points)[0])
        if indx_sampling< indx_sampling_old:
            continue
        state_update[indx_sampling_old:indx_sampling, :] = np.tile(state,(indx_sampling-indx_sampling_old,1))
        indx_sampling_old = indx_sampling
    state_update = state_update.astype('float16')  
    return [state_update, polII_pos]


def time_switching_between_on_off1_off2(time_points, ON_state, OFF1_state, OFF2_state, Promoter0):

    
    on_events = np.diff(ON_state)  # time where we switched to on
    if sum(ON_state) ==0:
        on_intervals = []
    else:
        start_on = np.where(on_events==1)[0]+1
        end_on = np.where(on_events == -1)[0]+1
        start_on = np.array([1])
        if end_on[0]< start_on[0]:
            start_on = np.insert(start_on,0, 0)
        on_intervals = np.array([start_on, end_on]).T.tolist()
        
    off1_events = np.diff(OFF1_state)  # time where we switched to  off1
    if sum(OFF1_state) == 0 :
        off1_intervals = []
    else:
        start_off1 = np.where(off1_events==1)[0]+1
        end_off1 = np.where(off1_events == -1)[0]+1 
        if sum(start_off1)==0 and sum(end_off1) !=0:
            start_off1 = np.array([1])
        if end_off1[0] < start_off1[0]:
            start_off1 =  np.unique(np.insert(start_off1,0, 0))
        off1_intervals = np.array([start_off1, end_off1], dtype = object).T.tolist()
    
    off2_events = np.diff(OFF2_state)  # time where we switched to  off2
    if sum(OFF2_state)==0:
        off2_intervals = []
    else:
        start_off2 = np.where(off2_events==1)[0] +1
        end_off2 = np.where(off2_events == -1)[0]+1
        if sum(start_off2)==0 and sum(end_off2) !=0:
            start_off2 = np.array([1])
        if end_off2[0] < start_off2[0]:
            start_off2 =  np.unique(np.insert(start_off2,0, 0))
        off2_intervals = np.array([start_off2, end_off2]).T.tolist()
        
    intervals_of_time_of_gene_state = np.array(on_intervals+off1_intervals + off2_intervals)
    state_of_gene_in_interval = np.array(len(on_intervals)*[0]+ len(off1_intervals)*[1]+ len(off2_intervals)*[2])
    if np.sum(intervals_of_time_of_gene_state)!=0:      
        sorting_indx = np.argsort(intervals_of_time_of_gene_state[:, 0])
        intervals_of_time_of_gene_state = intervals_of_time_of_gene_state[sorting_indx]
        state_of_gene_in_interval = state_of_gene_in_interval[sorting_indx]

    return intervals_of_time_of_gene_state, state_of_gene_in_interval
