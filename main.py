
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:43:27 2024

@author: mdouaihy
"""
import numpy as np
import sys
sys.path.append('./utilities/')
sys.path.append('../code_bcpd/utilities/')
from movies_combining import movies_combining_rawData
from ponxkini import ponxkini_deconvolveData,  ponxkini
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from func_simulations_prot_bcpd import simulating_mrna_prot_parallel
import pandas as pd
import os
from joblib import Parallel, delayed
from smt.sampling_methods import LHS
import re
import seaborn as sns
import joypy
plt.close('all')

inputdata = './Experimental Data/snaCRISPR_MS2_nc14/'
outputpath =  './Simulated Data/'
output_fig_final = './FIGURES_29012923_k3/'


if not os.path.exists(outputpath):
    os.mkdir(outputpath)    
    

if not os.path.exists(output_fig_final):
    os.mkdir(output_fig_final)    
    
percentage = 0.8 # probability of cut off points of bcpd  to be valid
kernel = 1 # smoothing of the data before running bcpd. kernel = 1 means no smoothing
filtered = 1 # filter nuclei after bcpd cut point based on criteria mentioned in the supp 
number_of_workers = 4 # number of workers for parallel computing 
nbr_benchmarking = 4 # number of samples for {nhk, thk, k2}

w_ponxkini = 8 # number of frames for for ponxkini analysis






""" switching parameters """
### mRNA information
kini = 0.35105 # 0.241687564*60 ##### at the maximum when no inhibitor
### prmoter switching information
km1 = 0.007 # 0.0041385181*60 ##### at the minimum when no inhibitor
kp1_act = 0.011 # 0.041687564*60 ##### at the maximum when no inhibitor 0.027363828000842 # 
kp1_rep = 0.001
kp2_rep = 0.044 # taken from nc14 bocpd cuttogff ### at min with inhibitor  
km2_act= 0.00 # 0.0041385181*60 ##### at the minimum when no inhibitor
km2_rep= 0.1125  # 0.25# #minkm*8 #### at max with inhibitor 





""" Latin Hypercube sampling """
xlimits = np.array([[4.4, 7.5],[7,20], [kp2_rep, 0.09]]) 
sampling = LHS(xlimits=xlimits)

x = sampling(nbr_benchmarking)
thk_list = x[:,0] 
nhk_list = x[:,1] 
kp2_act_list = x[:,2]




""" Hyperparameters for signal construction based on MS2 of the data"""

retention = 0
Polym_speed = 25 
TaillePreMarq = 1640
TailleSeqMarq = 1292 
TaillePostMarq = 1312 + Polym_speed*retention
EspaceInterPolyMin = 30
Intensity_for_1_Polym=1 
DureeSignal = (TaillePreMarq + TailleSeqMarq + TaillePostMarq) / Polym_speed;
    
tinter=(EspaceInterPolyMin/Polym_speed)
FreqEchSimu = 1/(EspaceInterPolyMin/Polym_speed)





""" load real first hitting time """
T0_path = inputdata + 'rawData/'
FilePath = T0_path + 'npzFile/'
xlsPath = T0_path + 'xlsFile/'
calibrtion = 1
extension = '.xlsx'
[FrameLen, dataExp, tmax_combined, tstart] = movies_combining_rawData(FilePath, xlsPath, calibrtion, extension)

files_nc14=np.array(os.listdir(xlsPath))
files_nc14 = list(map(lambda x: x.replace(extension,'') ,files_nc14))

T0_list = np.argmax(dataExp != 0, axis=0)*FrameLen
nsample = dataExp.shape[1]



""" load distribution of repression time from real data """

bcpd_fname = inputdata  + 'BCPD_Output/CP_0.6xMax_K_2/result_BCPD_filtered_1.npz'
content = np.load(bcpd_fname)
T0_0_true = content['T0_repressed']
DataExp_repressed = content['DataExp_repressed']
DataExp_activation = content['DataExp_activation']

check_nan_columns = np.where(np.nansum(DataExp_repressed==0, axis =0)+np.nansum(np.isnan(DataExp_repressed), axis = 0)==DataExp_repressed.shape[0])[0] # this checks nuclei that are full with 0 and nan
sd=DataExp_repressed.shape
frame_num=sd[0] ### number of frames
DureeSimu = frame_num*FrameLen  ### film duration in s
DureeAnalysee = DureeSignal + DureeSimu
num_possible_poly = round(DureeAnalysee/(EspaceInterPolyMin/Polym_speed)) # maximal number of polymerase positions
FreqEchImg = (1/FrameLen)
area = FreqEchImg/Polym_speed*Intensity_for_1_Polym*(TaillePostMarq+TailleSeqMarq/2)
DataExpSmooth = np.minimum(np.round(np.nansum(DataExp_repressed,axis = 0) / area), num_possible_poly)
check_less_than_pol_II = np.where(DataExpSmooth==0)[0]
delete_elemts_1 = np.unique(np.append(check_nan_columns,check_less_than_pol_II))
T0_1_true = np.delete(T0_0_true, delete_elemts_1)






""" additional hyperparameters"""
finaltime  = dataExp.shape[0]*FrameLen  # in seconds

time_points = np.arange(0,finaltime , FrameLen)     
frame_num=round(finaltime/FrameLen)

lframes = np.round(finaltime/FrameLen)
DureeSimu = lframes*FrameLen

DureeAnalysee = DureeSignal + DureeSimu
num_possible_poly = round(DureeAnalysee/(EspaceInterPolyMin/Polym_speed))

    
    

""" load protein data """

fname = inputdata + 'protein/snailLlama_nc14_in-patternEnrichment.xlsx'
protdata_table = pd.read_excel(fname)
tprot = protdata_table.iloc[1:,0].tolist()    

protConcent = np.array(protdata_table.iloc[1:,1:].values)
prot_mean = np.nanmean(protConcent, axis=1)
prot_interp = interp1d(np.array(tprot)*60, prot_mean, fill_value='extrapolate')(time_points)#(TTpol_nc14-dwell_time)





""" generate artificial data for comparison """
pars_prefixed=np.array([kini, kp1_act, kp1_rep, km1, 
                kp2_rep,
               km2_act, km2_rep
               ])    

fParam_prefixed =  outputpath + 'PreFixed_drosoParameters.npz'

np.savez(fParam_prefixed, 
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
          pars_prefixed = pars_prefixed
        )




""" ponx kinireal data"""
[TT_real, TTpol_real, kini_pred_real, num_gaps_real, sum_pon_est_real, sum_wt_2_est_real, mw_nuclei, sum_wt_2_nuclei_real, nintervals_nuclei] = ponxkini(w_ponxkini, extension, 'nc14',
     '', inputdata, '',fParam_prefixed,  '0', 0,'0', 10000)


TTpol_real = TTpol_real +  (TaillePreMarq + TailleSeqMarq+ TaillePostMarq)/Polym_speed/60
mean_wt = sum_pon_est_real/num_gaps_real # mean 
pon_est_real=1/mean_wt # inverse is pon kini


""" ponxkini bounds """

fname = inputdata + 'Hill_fitting/ponxkini_w' + str(w_ponxkini) +'_protx_1.xlsx'
ponxkini_plotted = pd.read_excel(fname, 'plotted value')
prot_nc14 = ponxkini_plotted['interpolated prot']
ponxkini_data = ponxkini_plotted['data ponxkini']
ponxkini_fitted = ponxkini_plotted['fitted ponxkini']
sheet_names = pd.ExcelFile(fname).sheet_names[4:]


CI_upper_bound_all = []
CI_lower_bound_all = []
for i in range(len(sheet_names)):
    bounds_sheet = pd.read_excel(fname, sheet_names[i])
    CI_upper_bound_all.append(bounds_sheet['Upper Bound'].to_numpy())
    CI_lower_bound_all.append( bounds_sheet['Lower Bound'].to_numpy())

CI_lower_bound = np.nanmax(np.array(CI_upper_bound_all), axis=0)
CI_upper_bound = np.nanmin(np.array(CI_lower_bound_all), axis=0)








""" launch simulations """
result = Parallel(n_jobs=number_of_workers,prefer="threads")(delayed(simulating_mrna_prot_parallel)(iexp, time_points, nsample, 
                  frame_num, finaltime, percentage, kernel, filtered, tprot, protConcent, T0_list,  nhk_list, thk_list, kp2_act_list, fParam_prefixed, outputpath) for iexp in range(nbr_benchmarking))





list_folder = [folder_i for folder_i in os.listdir(outputpath) 
               if 'prot_real' in folder_i and 
                  'npzFile' in os.listdir(os.path.join(outputpath, folder_i)) and 
                  'simulation_output.npz' in os.listdir(os.path.join(outputpath, folder_i, 'npzFile'))]



nhk_thk =  [re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", folder_i) for folder_i in list_folder]
nhk_thk = np.array([[float(number) for number in numbers] for numbers in nhk_thk])

nhk_list = nhk_thk[:,0]
thk_list = nhk_thk[:,1]
kp2_act_list = nhk_thk[:,3]

folder_i = 0
file_name = 'prot_real_n_' + str(nhk_list[folder_i]) + '_theta_' + str(thk_list[folder_i]) + '_kp2act_' + str(kp2_act_list[folder_i]) + '/'
dataPath = outputpath + file_name + 'npzFile/'

fname = dataPath + 'simulation_output.npz'
content = np.load(fname)
sd = np.array([400,658]) #content['DataExp'].shape
DataExp_all = np.zeros((len(thk_list), sd[0],sd[1])) 
DataExp_all[:] = np.nan


T0_0_all = np.zeros((len(thk_list), sd[1]))
T0_0_all[:] = np.nan
T0_1_all = np.zeros((len(thk_list),  sd[1]))
T0_1_all[:] = np.nan


pon_estexp_all = np.zeros((len(thk_list), len(pon_est_real)))
pon_estexp_all[:] = np.nan



for folder_i in range(len(thk_list)):
    print(folder_i)
    file_name = 'prot_real_n_' + str(nhk_list[folder_i]) + '_theta_' + str(thk_list[folder_i])  + '_kp2act_' + str(kp2_act_list[folder_i]) + '/'
    dataPath = outputpath + file_name + 'npzFile/'
    
    fname = dataPath + 'simulation_output.npz'    
    content = np.load(fname)
    DataExp = content['DataExp']
    DataExp_all[folder_i, :,:DataExp.shape[1]] = DataExp[:,:sd[1]]
    
    """ Time into repression """
    
    bcpd_fname  =  outputpath + file_name + '/CP_' + str(percentage) + 'xMax_K_' + str(kernel) + '/result_BCPD_filtered_' + str(filtered) + '.npz'
    content = np.load(bcpd_fname)
    T0_0 = content['T0_repressed']
    T0_1 = content['T0_1']
    
    T0_0_all[folder_i,:len(T0_0)] = T0_0
    T0_1_all[folder_i,:len(T0_1)] = T0_1

    
    """ ponxkini   """
    
    fname = dataPath + 'simulation_output.npz'    
    content = np.load(fname)            

    if 'pon_estexp' in content.files:
        pon_estexp = content['pon_estexp']
        TTpolexp = content['TTpolexp']
        
    else:
        
        [TTexp, TTpolexp, kini_predexp, num_gapsexp, sum_pon_estexp, sum_wt_2_estexp, mw_nuclei, sum_wt_2_nucleiexp, nintervals_nuclei] = ponxkini_deconvolveData(10000, w_ponxkini, 'nc14', '', fname, '', fParam_prefixed,'0', 0, filtered = '0')
        
        TTpolexp = TTpolexp +  (TaillePreMarq + TailleSeqMarq+ TaillePostMarq)/Polym_speed/60
        mean_wt = sum_pon_estexp/num_gapsexp # mean 
        pon_estexp=1/mean_wt # inverse is pon kini
        np.savez(fname,
                 DataExp = content['DataExp'],
                 mrna_nascent_output = content['mrna_nascent_output'],
                 T0_repressed = content['T0_repressed'],
                 time_points = content['time_points'],
                 PosPred = content['PosPred'],
                 DataPred = content['DataPred'],
                 pon_estexp = pon_estexp,
                 TTpolexp=  TTpolexp,
                 sum_pon_estexp = sum_pon_estexp,
                 num_gapsexp = num_gapsexp)
                
    pon_estexp_all[folder_i,:] = pon_estexp[:len(pon_est_real)] 


obj_ponxkini_all = np.zeros((len(thk_list),)) ### sum error squared of ponxkini data and ponxkini simulations

for fi in range(len(thk_list)):          
    obj_ponxkini_all[fi] = np.nansum((pon_estexp_all[fi,:]-pon_est_real)**2)   
    

ind_min_ponxkini =  np.argmin(obj_ponxkini_all)


""" figure 6 C, C' """
h, ax = plt.subplots(2,1, figsize=(5,6))

ind_min_ponxkini = ind_min_ponxkini
k_OFF1 = km1
k_ON1 = kp1_act + (kp1_rep-kp1_act)*(1 / (1 + (thk_list[ind_min_ponxkini] / prot_interp) ** nhk_list[ind_min_ponxkini]))
k_ON2 = kp2_act_list[ind_min_ponxkini] + (kp2_rep-kp2_act_list[ind_min_ponxkini])*(1 / (1 + (thk_list[ind_min_ponxkini] / prot_interp) ** nhk_list[ind_min_ponxkini]))
k_OFF2 = km2_act + (km2_rep-km2_act)*(1 / (1 + (thk_list[ind_min_ponxkini] / prot_interp) ** nhk_list[ind_min_ponxkini]))


ax[0].plot(time_points/60, prot_interp, linewidth = 3, label = 'protein')
ax[0].set_xlabel('Time (min)')
ax[0].set_ylabel('sna Concentration')


ax[1].plot(time_points/60, k_OFF1*np.ones((len(time_points))),linewidth = 3,  c = '#226258', label = r'$k_1^m$')
ax[1].plot(time_points/60, k_ON1*np.ones((len(time_points))), linewidth = 3, c = '#5A7B29', label = r'$k_1^p$')
ax[1].plot(time_points/60, k_OFF2,linewidth = 3,  c='#957A2A', label = r'$k_2^m$')
ax[1].plot(time_points/60, k_ON2*np.ones((len(time_points))), linewidth = 3, c= '#8A4F21', label = r'$k_2^p$')
ax[1].legend(frameon=False)
ax[1].set_xlabel('Time (min)')
ax[1].set_ylabel('Switching Parameters')
plt.tight_layout()
h.savefig(output_fig_final + 'fig_6C_model_intro.pdf')


writer = pd.ExcelWriter(output_fig_final + '/fig_6C_model_intro.xlsx', engine='xlsxwriter')
pdframe = pd.DataFrame({'Time':time_points/60, 'protein interpolated': prot_interp,
                          'km1':k_OFF1*np.ones((len(time_points))),
                          'kp1':k_ON1,
                          'km2':k_OFF2,
                          'kp2': k_ON2})
pdframe.to_excel(writer, sheet_name= 'sheet1', index=False, startrow = 0, startcol = 0)
writer.save()




""" figure 6-D"""

h =plt.figure(figsize=(6, 4))
plt.plot(prot_nc14,ponxkini_data,'x', label = 'Data')
plt.plot(prot_nc14, pon_estexp_all[ind_min_ponxkini,:],'r', linewidth = 5, label = 'Simulated ponxkini')
plt.fill_between(prot_nc14, CI_lower_bound, CI_upper_bound,  color='b', alpha=0.1)
plt.plot(prot_nc14,ponxkini_fitted ,'k',linewidth=5, label = 'Fitted ponxkini from real Data')
plt.xlabel('protein',fontsize=18)
plt.ylabel(r'$p_{on} \times k_{ini}$',fontsize=18)
plt.xlim([0,9])
plt.legend()
plt.title('obj w.r.t. ponxkini distribution')
plt.tight_layout()
h.savefig(output_fig_final + '/fig_6D_ponxkini.pdf')



writer = pd.ExcelWriter(output_fig_final + '/fig_6D_ponxkini.xlsx', engine='xlsxwriter')
pdframe = pd.DataFrame({'prot real data':prot_nc14, 
                        'ponxkini real data': pon_est_real,
                          'ponxkini lower bound real data':CI_upper_bound,
                          'ponxkini upper bound real data':CI_lower_bound,
                          'ponxkini fitted':ponxkini_fitted,
                          'ponxkini simulated':pon_estexp_all[ind_min_ponxkini,:]})
pdframe.to_excel(writer, sheet_name= 'sheet1', index=False, startrow = 0, startcol = 0)
writer.save()




""" figure 6 E """
fig = plt.figure( figsize=(6,4))

sns.histplot(T0_1_true / 60, bins=76, kde=False, color='blue', label='Real Histogram', stat='density')
sns.distplot(T0_1_all[ind_min_ponxkini, :], hist=False, kde=True,
             kde_kws={'shade': True, 'linewidth': 3}, color='orange',
             label='t0 1')
plt.axvline(np.median(T0_1_true / 60), color='blue', linestyle='dashed', linewidth=2, label='Median Real')
plt.axvline(np.nanmedian(T0_1_all[ind_min_ponxkini, :]), color='orange', linestyle='dashed', linewidth=2, label='Median Simulated')
plt.xlabel('Values (in minutes)')
plt.ylabel('Density')
plt.legend()
plt.title('obj w.r.t. ponxkini distribution')

plt.tight_layout()
fig.savefig(output_fig_final + '/fig_6E_repression_time_hist.pdf')

writer = pd.ExcelWriter(output_fig_final + '/fig_6E_repression_time_hist.xlsx', engine='xlsxwriter')
pdframe = pd.DataFrame({'time into repression filtered real data':T0_1_true / 60} )
pdframe.to_excel(writer, sheet_name= 'sheet1', index=False, startrow = 0, startcol = 0)
pdframe = pd.DataFrame({'kernel density estimation simulation': T0_1_all[ind_min_ponxkini, :]})
pdframe.to_excel(writer, sheet_name= 'sheet1', index=False, startrow = 0, startcol = 1)
pdframe = pd.DataFrame({
    'median ponxkini real data': [np.median(T0_1_true / 60)],
    'median ponxkini simulations': [np.nanmedian(T0_1_all[ind_min_ponxkini, :])]
})
pdframe.to_excel(writer, sheet_name='sheet1', index=False, startrow=0, startcol=2)
writer.save()





""" figure 6 F """

nbr_bins_t = 60

data = T0_1_true/60
hist, edges = np.histogram(data, bins=nbr_bins_t, density=True)
max_bin_index = np.argmax(hist)
bin_width = edges[1] - edges[0]
half_max_height = hist[max_bin_index] / 2
indices_half_width = np.where(hist > half_max_height)[0]
half_width_bins = indices_half_width[-1] - indices_half_width[0]
half_width_real = half_width_bins * bin_width



h = plt.figure( figsize=(5,6))
plt.hist(data, bins=nbr_bins_t, density=True, alpha=0.7, label='real data')
plt.axhline(half_max_height, color='red', linestyle='--', label='Half Height')
plt.axvline(edges[indices_half_width[0]], color='green', linestyle='--', label='Start of Half Width')
plt.axvline(edges[indices_half_width[-1]], color='blue', linestyle='--', label='End of Half Width')
plt.title("Half-Height Width :"+ str(np.round(half_width_real,3)))
plt.legend()
h.savefig(output_fig_final + '/figure_6F_example_half_hight_width_real_data.pdf')




writer = pd.ExcelWriter(output_fig_final + '/figure_6F_example_half_hight_width_real_data.xlsx', engine='xlsxwriter')
pdframe = pd.DataFrame({'real data to plot as hist':data})
pdframe.to_excel(writer, sheet_name= 'sheet1', index=False, startrow = 0, startcol = 0)
pdframe = pd.DataFrame({
    'half height': [half_max_height],
    'Start of Half Width': [edges[indices_half_width[0]]],
    'End of Half Width': [edges[indices_half_width[-1]]],
    'Half-Height Width': [half_width_real]})

pdframe.to_excel(writer, sheet_name= 'sheet1', index=False, startrow = 0, startcol = 1)
writer.save()




""" figure 6 G """
half_width_data_all = np.zeros((len(thk_list),))
hist_data_all = np.zeros((len(thk_list), nbr_bins_t))
for folder_i in range(len(thk_list)):
        
    data = T0_1_all[folder_i, ~np.isnan(T0_1_all[folder_i, :])]
    hist, edges = np.histogram(data, bins=nbr_bins_t, density=True)
    hist_data_all[folder_i,:]  = hist
    max_bin_index = np.argmax(hist)
    bin_width = edges[1] - edges[0]
    half_max_height = hist[max_bin_index] / 2
    indices_half_width = np.where(hist > half_max_height)[0]
    half_width_bins = indices_half_width[-1] - indices_half_width[0]
    half_width_data_all[folder_i] = half_width_bins * bin_width
    



indices_nhk = {index for index, value in enumerate(nhk_thk[:,0]) if 7 < value < 20}
indices_thk = {index for index, value in enumerate(nhk_thk[:,1]) if 4.4 < value < 7.5}
common_indices = list(indices_thk.intersection(indices_nhk))
nhk_list_restricted = nhk_list[common_indices]
thk_list_restricted = thk_list[common_indices]
list_folder_restricted = np.array(list_folder)[common_indices]



fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(thk_list[list(common_indices)], nhk_list[list(common_indices)], kp2_act_list[list(common_indices)], c=np.abs(half_width_data_all[list(common_indices)]), cmap='viridis', marker='o')

# Customize the plot
ax.set_xlabel('thk')
ax.set_ylabel('nhk')
ax.set_zlabel('kp2 act')
ax.set_title('half width')

# Add a color bar
cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0.1)
fig.savefig(output_fig_final + '/scan_thk_theta_kp2act_half_height_width.pdf')


fig.savefig(output_fig_final + '/scan_thk_theta_kp2_half_height_width.pdf')

writer = pd.ExcelWriter(output_fig_final + '/scan_thk_theta_kp2act_half_height_width.xlsx', engine='xlsxwriter')
pdframe = pd.DataFrame({'theta values ':thk_list[list(common_indices)],
                        'nhk values': nhk_list[list(common_indices)],
                        'kp2 act values': kp2_act_list[list(common_indices)],
                        'half height width values': half_width_data_all[list(common_indices)]
                        })
pdframe.to_excel(writer, sheet_name= 'sheet1', index=False, startrow = 0, startcol = 0)
writer.save()


kp2_rest = list({index for index, value in enumerate(kp2_act_list) if 0.06 <= value <= 0.065})

fig = plt.figure(figsize=(6,7), dpi=100)

scatter1 = plt.scatter(thk_list[kp2_rest], nhk_list[kp2_rest], c=np.abs(half_width_data_all)[kp2_rest], cmap='viridis', marker='s', s=100)
plt.title('half width', fontsize=10)
plt.xlabel("thk", fontsize=10)
plt.ylabel("nhk", fontsize=10)
plt.colorbar(scatter1)

fig.savefig(output_fig_final + '/fig_6G_scan_th_theta_kp2_0.06.pdf')


writer = pd.ExcelWriter(output_fig_final + '/fig_6G_scan_th_theta_kp2_0.06.xlsx', engine='xlsxwriter')
pdframe = pd.DataFrame({'theta values ':thk_list[list(kp2_rest)],
                        'nhk values': nhk_list[list(kp2_rest)],
                        'kp2 act values': kp2_act_list[list(kp2_rest)],
                        'half height width values': half_width_data_all[list(kp2_rest)]
                        })
pdframe.to_excel(writer, sheet_name= 'sheet1', index=False, startrow = 0, startcol = 0)
writer.save()


""" restriction on nh FIgure 6 H """
indices_nh = list({index for index, value in enumerate(nhk_list) if 17.30 <= value < 17.4})
indices_sort = np.array(indices_nh)[np.argsort( thk_list[indices_nh])][2:]
df = pd.DataFrame(T0_1_all[indices_sort, :].T)
df.columns = ['theta : '+ str(np.round(i,3)) for i in thk_list[indices_sort]]
df = df.dropna()
palette = sns.color_palette("viridis", len(df.columns))
fig, ax = joypy.joyplot(df, title = "Density of filtered T0 w.r.t. nh = 17.32", color = palette)
fig.savefig(output_fig_final + '/Fig_6H_scan_nh_17.32.pdf')


writer = pd.ExcelWriter(output_fig_final + '/Fig_6H_scan_nh_17.32.xlsx', engine='xlsxwriter')

df.to_excel(writer, sheet_name= 'sheet1', index=False, startrow = 0, startcol = 0)
writer.save()



""" restriction on theta Figure 6 I """
indices_theta = list({index for index, value in enumerate(nhk_thk[:,1]) if 4.0 <= value < 4.07})
indices_sort = np.array(indices_theta)[np.argsort( nhk_list[indices_theta])]
df = pd.DataFrame(T0_1_all[indices_sort, :].T)
df.columns = ['nh : '+ str(np.round(i,3)) for i in nhk_list[indices_sort]]
df = df.dropna()
palette = sns.color_palette("viridis", len(df.columns))
fig, ax = joypy.joyplot(df, title = "Density of filtered T0 w.r.t. theta = 4", color = palette)
fig.savefig(output_fig_final + '/fig_6I_scan_nh_theta_4.pdf')


writer = pd.ExcelWriter(output_fig_final + '/fig_6I_scan_nh_theta_4.xlsx', engine='xlsxwriter')

df.to_excel(writer, sheet_name= 'sheet1', index=False, startrow = 0, startcol = 0)
writer.save()
