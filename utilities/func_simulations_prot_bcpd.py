#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:03:52 2024


@author: mdouaihy
"""


from utilities_artificial_data import gillespie_main_prot_real, propensities_feedback_negative_3S_k2m_prot
import numpy as np
from signal_construction import sumSignal1_par_gillespie
from scipy.interpolate import interp1d
from MLE_gamma import MLE_gamma
from GaussianUnknownMeanAndVariance import GaussianUnknownMeanUnknownVariance
from bcpd_attribute import bocd
from retriving_time_points import retriving_time_100723
import os 


def simulating_mrna_prot_parallel(iexp, time_points, nsample, frame_num, finaltime, percentage, kernel, filtered, tprot, protdata, T0_list,  nhk_list, thk_list, kp2_act_list, fParam_prefixed, outputpath):

    nhk2 = nhk_list[iexp]
    thk2 = thk_list[iexp]
    kp2_act = kp2_act_list[iexp]
    ### Parameters depending on the gene

    file_name = 'prot_real_n_' + str(nhk2) + '_theta_' + str(thk2) + '_kp2act_' + str(kp2_act) + '/'
    
    outputPath = outputpath + file_name 

    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
        
    
    content = np.load(fParam_prefixed)

    Polym_speed = content['Polym_speed']  
    TaillePreMarq = content['TaillePreMarq']
    TailleSeqMarq = content['TailleSeqMarq']
    TaillePostMarq = content['TaillePostMarq']
    EspaceInterPolyMin = content['EspaceInterPolyMin']
    FrameLen = content['FrameLen']
    Intensity_for_1_Polym = content['Intensity_for_1_Polym']
    FreqEchImg = content['FreqEchImg']
    DureeSignal = content['DureeSignal']
    FreqEchSimu = content['FreqEchSimu']
    retention = content['retention']
    pars_prefixed = content['pars_prefixed']

    [kini, kp1_act, kp1_rep, km1, kp2_rep, km2_act, km2_rep] = pars_prefixed
    
    pars=np.array([kini, kp1_act, kp1_rep, km1, kp2_act, kp2_rep,
                       km2_act, km2_rep, nhk2, thk2
                       ])
        
              
    
    fParam =  outputPath + 'drosoParameters.npz'
    #
    np.savez(fParam, 
              Polym_speed = Polym_speed,  
              TaillePreMarq = TaillePreMarq,
              TailleSeqMarq = TailleSeqMarq,
              TaillePostMarq = TaillePostMarq,
              EspaceInterPolyMin = EspaceInterPolyMin,
              FrameLen = FrameLen,
              Intensity_for_1_Polym = Intensity_for_1_Polym,
              FreqEchImg = FreqEchImg,
              DureeSignal = DureeSignal,
              FreqEchSimu = FreqEchSimu,
              retention = retention,
              pars = pars
            )
    
    
    DataExp, mrna_nascent_output, T0_repressed, PosPred = simulating_mrna_prot(time_points, tprot, protdata, T0_list+ TaillePreMarq/Polym_speed, nsample, frame_num, finaltime, nhk2, thk2, percentage, kernel, filtered, outputPath, fParam)

    dataPath =  outputPath + 'npzFile/'
    if not os.path.exists(dataPath):
        os.mkdir(dataPath)    
    
    fParam = dataPath + 'simulation_output.npz'    
    np.savez(fParam,
             DataExp = DataExp.astype('float16'),
             mrna_nascent_output = mrna_nascent_output.astype('float16'),
             T0_repressed = T0_repressed.astype('float16'),
             time_points = time_points.astype('float16'),
             PosPred = PosPred.astype('float16'),
             DataPred = DataExp.astype('float16'))    
    
    
    return [DataExp, mrna_nascent_output, T0_repressed]


    
    



def simulating_mrna_prot(time_points, tprot, protdata, T0_sampled, nsample, frame_num, finaltime, nhk2, thk2, percentage, kernel, filtered, outputPath, fParam):
        

    bcpd_outputpath = outputPath + '/CP_' + str(percentage) + 'xMax_K_' + str(kernel) + '/'


    if not os.path.exists(bcpd_outputpath):
        os.mkdir(bcpd_outputpath)
        

    """ Hyperparameters used for initiating Gillespie """
    
    ### State 
    nascentMRNA = 0
    Promoter0 = 'ON' # can take the form ON, OFF1, OFF2 since it's a 3 state
    
    
    """ protein data information """
    prot_mean = np.nanmean(protdata, axis=1)
    prot_interp = interp1d(np.array(tprot)*60, prot_mean, fill_value='extrapolate')(time_points)#(TTpol_nc14-dwell_time)

            
    ### Hyperparameters used for initiating Gillespie
    
    if Promoter0 == 'OFF1':
        state = [0,1,0,nascentMRNA] ### start in state OFF,   
    elif Promoter0 == 'OFF2':
        state = [0, 0,1,nascentMRNA]
    elif Promoter0 =='ON':
        state = [1,0,0,nascentMRNA]
        

    fcontent =  np.load(fParam)
    

    FrameLen = fcontent['FrameLen']
    pars = fcontent['pars']
    FrameLen = fcontent['FrameLen']
    EspaceInterPolyMin = fcontent['EspaceInterPolyMin']
    Polym_speed = fcontent['Polym_speed']
    DureeSignal = fcontent['DureeSignal']
    TailleSeqMarq = fcontent['TailleSeqMarq']
    TaillePostMarq = fcontent['TaillePostMarq']
    FreqEchImg = fcontent['FreqEchImg']
    Intensity_for_1_Polym = fcontent['Intensity_for_1_Polym']
    
    DureeSimu = frame_num*FrameLen
    DureeAnalysee = DureeSignal + DureeSimu 
    tinter=(EspaceInterPolyMin/Polym_speed)
    num_possible_poly = round(DureeAnalysee/(EspaceInterPolyMin/Polym_speed))
        
    [kini, kp1_act, kp1_rep, km1, kp2_act, kp2_rep, km2_act, km2_rep, nhkm2, thkm2] = pars
    mrna_nascent_output = np.zeros((len(time_points), nsample))
    DataExp = np.zeros((frame_num, nsample))   
    PosPred = np.zeros((num_possible_poly, nsample))
    

    for i in range(nsample):
        [state_update, polII_pos] = gillespie_main_prot_real(propensities_feedback_negative_3S_k2m_prot,state, pars, 'feedback_negative_3S_prot', time_points, num_possible_poly, tinter, int(T0_sampled[i]/FrameLen), prot_interp)
        
        mRNA_nascent = state_update[:,3]
    
        positions = np.where(polII_pos ==1)[0]
        signal_sampled = sumSignal1_par_gillespie(positions, fParam, frame_num)
            
        mrna_nascent_output[:,i] = mRNA_nascent
        DataExp[:,i] = signal_sampled
        PosPred[:,i] = polII_pos
        


    
    
    
    
    hazard = 1/10000  # Constant prior on changepoint probability.     
    n2 = DataExp.shape
    nexp = n2[1]
    frame_num = n2[0]
    
    
    max_intensity_EXP = np.nanmax(DataExp, axis = 0)
    T0_EXP = np.zeros((nexp,))
    for i in range(nexp):
        ihit_exp = np.where(DataExp[:,i] > max_intensity_EXP[i]/5)[0]
        if len(ihit_exp)== 0:
            ihit_exp=n2[0]
            T0_EXP[i] = (ihit_exp)/FreqEchImg 
        else:
            ihit_exp=min(np.where(DataExp[:,i]> max_intensity_EXP[i]/5)[0])
            T0_EXP[i] = (ihit_exp+1)/FreqEchImg 


    k0 = 1.5    # The prio
    mean0 = np.mean(DataExp[int(5*60/FrameLen):int(10*60/FrameLen),:])
    [alpha0, beta0] = MLE_gamma(np.var(DataExp[int(5*60/FrameLen):int(10*60/FrameLen),:],axis=0))
    
    
    set_parameters_exp = [mean0, k0, alpha0, beta0]
    
    p = 0.8
    
    
        
    ######## Data needed for bcpd
    DataExp_1st_part = np.full((DataExp.shape[0],DataExp.shape[1]),np.nan)
    DataExp_2nd_part = np.full((DataExp.shape[0],DataExp.shape[1]),np.nan)
    
    T0_repressed = []
    
    R_exp_list = []
    
    def smooth(y, box_pts): # the kernel is nomalized (box blur)
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    for data_i in range(nexp):
        
        data_exp = smooth(DataExp[:,data_i],kernel)
        set_parameters_exp = [mean0, alpha0, k0, beta0]
        
        model_exp          = GaussianUnknownMeanUnknownVariance(set_parameters_exp[0], set_parameters_exp[1], set_parameters_exp[2], set_parameters_exp[3])
        R_exp = bocd(data_exp, model_exp, hazard)
        [x_axis_lines, proba_line] = retriving_time_100723(R_exp,p)
        R_exp_list.append(R_exp)

        
        
        cond_for_cpd = np.where(data_exp>=np.nanmax(data_exp)*percentage)[0][-1] + 1
        cp_start_point = [x[0] for x in x_axis_lines]
        indx_bcpd_pt = np.where((cp_start_point-cond_for_cpd)>=0)[0]
        if len(indx_bcpd_pt) ==0:
            bcpd_pt = len(data_exp)
        else:
            indx_bcpd_pt = indx_bcpd_pt[0]
        
            bcpd_pt = x_axis_lines[indx_bcpd_pt][0]

        DataExp_1st_part[:bcpd_pt, data_i] = DataExp[ :bcpd_pt,data_i]
        DataExp_2nd_part[:DataExp.shape[0]-bcpd_pt, data_i] = DataExp[bcpd_pt:,data_i]
        
        
        T0_repressed.append(bcpd_pt)
    
 



    
    nan_rows = np.where(np.isnan(np.sum(DataExp_1st_part, axis=1)))[0]
    
    if len(nan_rows)==0:
        DataExp_0 = DataExp_1st_part
    else:
        DataExp_0 = DataExp_1st_part[:nan_rows[0],:]
    
    nan_rows = np.where(np.nansum(DataExp_2nd_part, axis=1)==0)[0]
            
    if len(nan_rows)==0:
        DataExp_1 = DataExp_2nd_part
    else:
        DataExp_1 = DataExp_2nd_part[:nan_rows[0],:]
    
    
    
    T0_repressed = np.array(T0_repressed)
    
    

    
    if filtered ==1:
    
        """ check points to eliminate nuclei in two steps:
            1- remove 0 nuclei
            2- remove nuclei that has less then 10% of max number of pol II per zone """
    
        check_nan_columns = np.where(np.sum(DataExp_0==0, axis =0)+np.sum(np.isnan(DataExp_0), axis = 0)==DataExp_0.shape[0])[0] # this checks nuclei that are full with 0 and nan
    
        
        """ remove nuclei with 0 pol II """
        sd=DataExp_0.shape
        frame_num=sd[0] ### number of frames
        DureeSimu = frame_num*FrameLen  ### film duration in s
        DureeAnalysee = DureeSignal + DureeSimu
        num_possible_poly = round(DureeAnalysee/(EspaceInterPolyMin/Polym_speed)) # maximal number of polymerase positions
        area = FreqEchImg/Polym_speed*Intensity_for_1_Polym*(TaillePostMarq+TailleSeqMarq/2)
        DataExpSmooth = np.minimum(np.round(np.sum(DataExp_0,axis = 0) / area), num_possible_poly)
        check_less_than_pol_II = np.where(DataExpSmooth==0)[0]
         
        delete_elemts_0 = np.unique(np.append(check_nan_columns,check_less_than_pol_II))
        DataExp_0 = np.delete(DataExp_0, delete_elemts_0, 1)

        check_nan_columns = np.where(np.sum(DataExp_1==0, axis =0)+np.sum(np.isnan(DataExp_1), axis = 0)==DataExp_1.shape[0])[0] # this checks nuclei that are full with 0 and nan
    
        
        """ remove nuclei with less than 1 pol II """
        sd=DataExp_1.shape
        frame_num=sd[0] ### number of frames
        DureeSimu = frame_num*FrameLen  ### film duration in s
        DureeAnalysee = DureeSignal + DureeSimu
        num_possible_poly = round(DureeAnalysee/(EspaceInterPolyMin/Polym_speed)) # maximal number of polymerase positions
        area = FreqEchImg/Polym_speed*Intensity_for_1_Polym*(TaillePostMarq+TailleSeqMarq/2)
        DataExpSmooth = np.minimum(np.round(np.sum(DataExp_1,axis = 0) / area), num_possible_poly)
        check_less_than_pol_II = np.where(DataExpSmooth==0)[0]
             
        delete_elemts_1 = np.unique(np.append(check_nan_columns,check_less_than_pol_II))

        DataExp_1 = np.delete(DataExp_1, delete_elemts_1, 1)
        



    np.savez(bcpd_outputpath + '/full_bcpd_results_p_' + str(p) + '.npz', 
             DataExp = DataExp.astype('float16'),
             R_exp_list = np.array(R_exp_list).astype('float16'))


    T0_0 = (T0_repressed-1)/FreqEchImg/60
    delete_elemts_1 = np.unique(np.append(check_nan_columns,check_less_than_pol_II))
    T0_1 = np.delete(T0_0, delete_elemts_1)
    
    
    np.savez(bcpd_outputpath + '/result_BCPD_filtered_' + str(filtered) + '.npz', 
             DataExp_activation = DataExp_0.astype('float16'),
             DataExp_repressed = DataExp_1.astype('float16'),
             DataExp = DataExp.astype('float16'),
             T0_repressed = (T0_repressed-1)/FreqEchImg,
             T0_1 = T0_1,
             PosPred = PosPred.astype('float16'),
             DataPred = DataExp.astype('float16'))
    


    return DataExp.astype('float16'), mrna_nascent_output.astype('float16'), T0_repressed.astype('float16'), PosPred.astype('float16')