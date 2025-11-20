# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 08:31:45 2018
'
@author: AGKremkow-europix set-up
"""
# %%
#import scipy.io as io
import numpy as np
#import OurSetup
import OurSetup_New as OurSetup
import matplotlib.pyplot as plt
import ImageStimulus
import MovingGratingStimulus
import MovingLocalGratingStimulus
import MovingBarStimulus
import MovingBarGridStimulus
import LoomingStimulus
import NaturalImageStimulusWithGray_CG

from time import time
import CSDStimulus    
from scipy.signal import chirp


# %%# %
start = time()
stimuli = ['mb','sl36x22_3','sd36x22_3','chi','mg','csd'] #['slquick'] ['slquick'] #

#['cm_36x22']#['mb_thin','sl36x22_05','sd36x22_05','sl36x22_01','sd36x22_01','chi_ultra_fast','mg','mb','sl36x22_2','sd36x22_2','sl36x22_1','sd36x22_1','chi','lo','cm_36x22','csd'] # ['slquick'] # ['mb','sl15x15_2','sd15x15_2','sl15x15_1','sd15x15_1','sl15x15_3','sd15x15_3','mg'] 
# ['slquick', 'sdquick','mg','mb','lo','chi','Kerstin_long','Kerstin_short','slquick','sdquick',] #['slquick'] # ['Kerstin short']
# Tatiana  ['slquick', 'sdquick','mb','mp']['slquick'] #['slquick'] #['slquick'] # 
#  ['sl10x10_1','sd10x10_1','sl10x10_2','sd10x10_2','mg','mb','mp','chi','lo','cm_10x10'] # ['mg'] # ['mb'] # ['slquick'] # ['mg','mb','mp','chi','lo','cm_36x22'] #  ['sdquick'] # ['mg','mb','mp'] #
 #  ['cm_10x10'] #  ['chi'] #['slquick'] #
# new stim Caro 'Images'
# ['mb','sl36x22_3','sd36x22_3','sl36x22_2','sd36x22_2','sl36x22_1','sd36x22_1','chi','lo','mg','mp','cm_36x22','mlg'] #
penetration = 'a11'
labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p'] # ['b'] #['b']#['o','p','q']# ['a'] #['a','b','c','d','e','f']#  ['a','b','c','d'] # ['b'] #  ['g','h','i','j','k','l'] 
#  ['a'] # ['b'] # ['test'] #['j','k','l'] # ['c','d','e','f'] # ['b'] # ['k','l','m'] 
# Kerstin 
#  ['g','h','i','j','k','l'] # 
#WARNING THIS COMING LINE SAVES THE STIM PARAMETERS LEAVE IT TO 0
test = 0 # if true nothing will be saved. This is good e.g. when the online ananlys computer is not on
#LEAVE ME TO ZERO!!!!!
closed_loop_with_analysis = 0 # 0 = no, 1 = yes

header = {}
header['date'] = '20230801'
header['name'] = 'Ababx_first_reco' # 'test' #'tryout' #'LGN_CG'#'test'  # 'SC_JS' #  SC_JS # LGN_CG
header['electrode'] = 'Neuropixel'#'conditionning' # 'poly3_subplots'#'linear'#'poly3_subplots'#'linear'##'poly3' 'linear'#'#'linear'#'linear'#'Poly3-32channels' #'Neuropixels'#'A1x32-5mm-50-177-A32' # 'A1x32-5mm-25-177-A32' '64 ch A8*8 empty' '64 ch A8*8'
header['n_chs'] = 384 

header['penetrationtype'] = 'Mouse_vertical'#'Neuropixel-above-insertion-25°' # 'conditionning' # 'from above: vertical retinotropy' # s 'from the back 20°' #'from the back 20°' # 'from above: horizontal A88 retinotropy'
#header['folder_tmp'] = u'\\\\ONLINEANALYSIS\\ex change-folder\\onlinetmp\\'
#header['folder_save'] = u'\\\\ONLINEANALYSIS\\exchange-folder\\onlinedata\\'
header['folder_tmp'] = u'C:\\Users\\AGKremkow\\Desktop\\Neuropixels\\onlinetmp\\'
header['folder_save'] = u'C:\\Users\\AGKremkow\\Desktop\\Neuropixels\\onlinedata\\'

monitor = {}
monitor['distance'] = 10
monitor['background'] = [0.,0.,0.]
monitor['refreshrate'] = 120.
monitor['type'] =  'Dell' #   'Dell'  #     
# position of the stimulus
position = np.array([0.,0.]) # negative: right and up, positive: left and down 
#### WARNING for the dome it is inverted! poisitve: right and up, negative: left and down 
# stimulus params
stimulus = {}
stimulus['onset_adaptation_time_sec'] = 1.

# analysis
analysis = {}
n_stimuli = len(stimuli)

#%% we open a window for all stimuli. then we might have to set the backgorund
win,trigger = OurSetup.OpenScreen(monitor['background'],monitor['distance'],monitor['type'])

setup = {}
setup['win'] = win
setup['trigger'] = trigger
 #%%
for i in range(n_stimuli):
    # %
    stimulus_type = stimuli[i]
    label = labels[i]
    
    experiment_name = penetration+label+stimulus_type
    
    stimulus['type'] = stimulus_type
    header['label'] = label
    header['penetration'] = penetration
    header['experiment_name'] = experiment_name
    
    # all parameters
    params = {}
    params['test'] = test
    params['closed_loop_with_analysis'] = closed_loop_with_analysis
    params['header'] = header
    params['monitor'] = monitor
    
    
    ################## search stimulus #############################
    if stimulus_type == 'slquick':
        # load the stimulus file
#        filename = 'locally_light_sparse_noise_45_25_target_size_3_targets_per_frame_2_trials_10_background_-1.0_20181020.npy' 
        filename = 'locally_light_sparse_noise_36_22_target_size_3_targets_per_frame_2_trials_10_background_-1.0_20181120.npy'#'locally_light_sparse_noise_45_25_target_size_3_targets_per_frame_2_trials_10_background_-1.0_20181020.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] =[background,background,background]# [1.,1.,1.]#
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 6
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
            
    elif stimulus_type == 'sdquick':
        # load the stimulus file
#        filename = 'locally_dark_sparse_noise_45_25_target_size_3_targets_per_frame_2_trials_10_background_1.0_20181020.npy'
        filename = 'locally_dark_sparse_noise_36_22_target_size_3_targets_per_frame_2_trials_10_background_1.0_20181120.npy'#
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 6
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        

    elif stimulus_type == 'mlg':
        stimulus['trials'] = 10
        stimulus['temporalfrequencies'] = np.array([3.])#,2.,3.]) # in Hz, maybe here going to 2 would save time? # for LGN: 2,3,4 Hz
        stimulus['spatialfrequencies'] = np.array([0.08])#([0.04,0.08]) #np.array([0.01,0.02,0.04,0.08,0.16]) # # np.array([0.02]) #np.array([0.01,0.02,0.04,0.08,0.16])
        stimulus['orientations'] = np.linspace(0.,360.-(360/12.),12)# np.linspace(0.,360.-(360/12.),12) #np.array([270.]) # np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        stimulus['size'] = np.array([20.,20.]) # ([400.,400.]) # to change in order to reduce the mask size
        stimulus['phase'] = 0.0
        stimulus['duration'] = 2.0
        stimulus['pause'] = 0.5
        stimulus['position'] = position
        stimulus['scale'] = 1.
        positions_x = [0.,22.5]
        positions_y = [0.,0.]
        stimulus['positions_x'] = positions_x
        stimulus['positions_y'] = positions_y
        
        
        params['monitor']['background'] = [0.,0.,0.]
        
        analysis['type'] = 'TuningPolar'
        analysis['parameters'] = ['orientations'] 
        analysis['psth_duration'] = stimulus['duration']
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        MovingLocalGratingStimulus.run_stimulus(params,setup)

#    ################## Spike Count Corr  ###########################
#  ################## sparse noise for the screen ###########################
        
    elif stimulus_type == 'sl45x25_1':
        # load the stimulus file
        filename = 'locally_light_sparse_noise_45_25_target_size_1_targets_per_frame_6_trials_50_background_-1.0_20181020.npy'#'locally_light_sparse_noise_36_18_target_size_1_targets_per_frame_4_trials_100_background_0.0_20181114.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        
    elif stimulus_type == 'sl45x25_1':
        # load the stimulus file
        filename = 'locally_dark_sparse_noise_45_25_target_size_1_targets_per_frame_6_trials_50_background_1.0_20181020.npy'#'locally_dark_sparse_noise_36_18_target_size_1_targets_per_frame_4_trials_100_background_-0.0_20181114.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        
    elif stimulus_type == 'sl45x25_2':
        # load the stimulus file
        filename = 'locally_light_sparse_noise_45_25_target_size_2_targets_per_frame_4_trials_30_background_-1.0_20181020.npy'#'locally_light_sparse_noise_36_18_target_size_2_targets_per_frame_4_trials_30_background_0.0_20181115.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        
    elif stimulus_type == 'sd45x25_2':
        # load the stimulus file
        filename = 'locally_dark_sparse_noise_45_25_target_size_2_targets_per_frame_4_trials_30_background_1.0_20181020.npy'#'locally_dark_sparse_noise_36_18_target_size_2_targets_per_frame_4_trials_30_background_-0.0_20181115.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
    
    elif stimulus_type == 'sl45x25_3':
        # load the stimulus file
        filename = 'locally_light_sparse_noise_45_25_target_size_3_targets_per_frame_2_trials_10_background_-1.0_20181020.npy'#'locally_light_sparse_noise_36_18_target_size_2_targets_per_frame_4_trials_30_background_0.0_20181115.npy'
                    
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        
    elif stimulus_type == 'sd45x25_3':
        # load the stimulus file
        filename = 'locally_dark_sparse_noise_45_25_target_size_3_targets_per_frame_2_trials_10_background_1.0_20181020.npy'#'locally_dark_sparse_noise_36_18_target_size_2_targets_per_frame_4_trials_30_background_-0.0_20181115.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        
    ################## Light sparse noise ###########################
    
    ################## sparse noise on sub-section of the screen ###########################
    elif stimulus_type == 'sl15x15_1':
        # load the stimulus file
        
        filename = 'locally_light_sparse_noise_15_15_target_size_1_targets_per_frame_1_trials_50_background_-1.0_20190625.npy'#'locally_light_sparse_noise_36_18_target_size_1_targets_per_frame_4_trials_100_background_0.0_20181114.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        
    elif stimulus_type == 'sl15x15_2':
        # load the stimulus file
        filename = 'locally_light_sparse_noise_15_15_target_size_2_targets_per_frame_1_trials_30_background_-1.0_20190625.npy'#'locally_light_sparse_noise_36_18_target_size_1_targets_per_frame_4_trials_100_background_0.0_20181114.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
    
    elif stimulus_type == 'sl15x15_3':
        # load the stimulus file
        filename = 'locally_light_sparse_noise_15_15_target_size_3_targets_per_frame_1_trials_10_background_-1.0_20190625.npy'#'locally_light_sparse_noise_36_18_target_size_1_targets_per_frame_4_trials_100_background_0.0_20181114.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
    
    elif stimulus_type == 'sd15x15_1':
        # load the stimulus file
        filename = 'locally_dark_sparse_noise_15_15_target_size_1_targets_per_frame_1_trials_50_background_1.0_20190625.npy'#'locally_light_sparse_noise_36_18_target_size_1_targets_per_frame_4_trials_100_background_0.0_20181114.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        
    elif stimulus_type == 'sd15x15_2':
        # load the stimulus file
        filename ='locally_dark_sparse_noise_15_15_target_size_2_targets_per_frame_1_trials_30_background_1.0_20190625.npy'#'locally_light_sparse_noise_36_18_target_size_1_targets_per_frame_4_trials_100_background_0.0_20181114.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        
    elif stimulus_type == 'sd15x15_3':
        # load the stimulus file
        filename = 'locally_dark_sparse_noise_15_15_target_size_3_targets_per_frame_1_trials_10_background_1.0_20190625.npy'#'locally_light_sparse_noise_36_18_target_size_1_targets_per_frame_4_trials_100_background_0.0_20181114.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)

        
  ################## sparse noise for the dome ###########################
        
    elif stimulus_type == 'sl36x22_1':
        # load the stimulus file
        filename = 'locally_light_sparse_noise_36_22_target_size_1_targets_per_frame_6_trials_50_background_-1.0_20181120.npy'#'locally_light_sparse_noise_36_18_target_size_1_targets_per_frame_4_trials_100_background_0.0_20181114.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        
    elif stimulus_type == 'sd36x22_1':
        # load the stimulus file
        filename = 'locally_dark_sparse_noise_36_22_target_size_1_targets_per_frame_6_trials_50_background_1.0_20181120.npy'#'locally_dark_sparse_noise_36_18_target_size_1_targets_per_frame_4_trials_100_background_-0.0_20181114.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)

    elif stimulus_type == 'sd36x22_05':
        # load the stimulus file
        filename = 'locally_dark_sparse_noise_36_22_target_size_1_targets_per_frame_6_trials_50_background_1.0_20181120.npy'#'locally_dark_sparse_noise_36_18_target_size_1_targets_per_frame_4_trials_100_background_-0.0_20181114.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 2.5 # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
    
    elif stimulus_type == 'sd36x22_01':
        # load the stimulus file
        filename = 'locally_dark_sparse_noise_36_22_target_size_1_targets_per_frame_6_trials_50_background_1.0_20181120.npy'#'locally_dark_sparse_noise_36_18_target_size_1_targets_per_frame_4_trials_100_background_-0.0_20181114.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 1. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        
        
    elif stimulus_type == 'sd140x80_1':
        # load the stimulus file
        filename = 'Sd_140_80_l_20200109.npy'#'locally_dark_sparse_noise_36_18_target_size_1_targets_per_frame_4_trials_100_background_-0.0_20181114.npy'
        stimulus['filename'] = filename
        stimulus_frames = np.load('stimuli/'+filename)
        #stimulus_frames = stimulus_tmp['frames']
        background = 1.#stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 1. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
    
    elif stimulus_type == 'sl140x80_1':
        # load the stimulus file
        filename = 'Sl_140_80_d_20200109.npy'#'locally_dark_sparse_noise_36_18_target_size_1_targets_per_frame_4_trials_100_background_-0.0_20181114.npy'
        stimulus['filename'] = filename
        stimulus_frames = np.load('stimuli/'+filename)
        #stimulus_frames = stimulus_tmp['frames']
        background = -1.#stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 1. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        
    elif stimulus_type == 'sl36x22_05':
        # load the stimulus file
        filename = 'locally_light_sparse_noise_36_22_target_size_1_targets_per_frame_6_trials_50_background_-1.0_20181120.npy'#'locally_light_sparse_noise_36_18_target_size_1_targets_per_frame_4_trials_100_background_0.0_20181114.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 2.5 # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
    
    elif stimulus_type == 'sl36x22_01':
        # load the stimulus file
        filename = 'locally_light_sparse_noise_36_22_target_size_1_targets_per_frame_6_trials_50_background_-1.0_20181120.npy'#'locally_light_sparse_noise_36_18_target_size_1_targets_per_frame_4_trials_100_background_0.0_20181114.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 1. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        
        
    elif stimulus_type == 'sl36x22_2':
        # load the stimulus file
        filename = 'locally_light_sparse_noise_36_22_target_size_2_targets_per_frame_4_trials_30_background_-1.0_20181120.npy'#'locally_light_sparse_noise_36_18_target_size_2_targets_per_frame_4_trials_30_background_0.0_20181115.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        
    elif stimulus_type == 'sd36x22_2':
        # load the stimulus file
        filename = 'locally_dark_sparse_noise_36_22_target_size_2_targets_per_frame_4_trials_30_background_1.0_20181120.npy'#'locally_dark_sparse_noise_36_18_target_size_2_targets_per_frame_4_trials_30_background_-0.0_20181115.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
    
    elif stimulus_type == 'sl36x22_3':
        # load the stimulus file
        filename = 'locally_light_sparse_noise_36_22_target_size_3_targets_per_frame_2_trials_10_background_-1.0_20181120.npy'#'locally_light_sparse_noise_36_18_target_size_2_targets_per_frame_4_trials_30_background_0.0_20181115.npy'
                    
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        
    elif stimulus_type == 'sd36x22_3':
        # load the stimulus file
        filename = 'locally_dark_sparse_noise_36_22_target_size_3_targets_per_frame_2_trials_10_background_1.0_20181120.npy'#'locally_dark_sparse_noise_36_18_target_size_2_targets_per_frame_4_trials_30_background_-0.0_20181115.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        
    elif stimulus_type == 'sl36x22_g_1':
        # load the stimulus file
        filename = 'locally_light_sparse_noise_36_22_target_size_1_targets_per_frame_6_trials_50_background_0.0_20181120.npy'#'locally_light_sparse_noise_36_18_target_size_1_targets_per_frame_4_trials_100_background_0.0_20181114.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        
    elif stimulus_type == 'sd36x22_g_1':
        # load the stimulus file
        filename = 'locally_dark_sparse_noise_36_22_target_size_1_targets_per_frame_6_trials_50_background_-0.0_20181120.npy'#'locally_dark_sparse_noise_36_18_target_size_1_targets_per_frame_4_trials_100_background_-0.0_20181114.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        
    elif stimulus_type == 'sl36x22_g_2':
        # load the stimulus file
        filename = 'locally_light_sparse_noise_36_22_target_size_2_targets_per_frame_4_trials_30_background_0.0_20181120.npy'#'locally_light_sparse_noise_36_18_target_size_2_targets_per_frame_4_trials_30_background_0.0_20181115.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        
    elif stimulus_type == 'sd36x22_g_2':
        # load the stimulus file
        filename = 'locally_dark_sparse_noise_36_22_target_size_2_targets_per_frame_4_trials_30_background_-0.0_20181120.npy'#'locally_dark_sparse_noise_36_18_target_size_2_targets_per_frame_4_trials_30_background_-0.0_20181115.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
    
    elif stimulus_type == 'sl36x22_g_3':
        # load the stimulus file
        filename = 'locally_light_sparse_noise_36_22_target_size_3_targets_per_frame_2_trials_10_background_0.0_20181120.npy'#'locally_light_sparse_noise_36_18_target_size_2_targets_per_frame_4_trials_30_background_0.0_20181115.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        
    elif stimulus_type == 'sd36x22_g_3':
        # load the stimulus file
        filename = 'locally_dark_sparse_noise_36_22_target_size_3_targets_per_frame_2_trials_10_background_-0.0_20181120.npy'#'locally_dark_sparse_noise_36_18_target_size_2_targets_per_frame_4_trials_30_background_-0.0_20181115.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        
    elif stimulus_type == 'sl72x44_4':
        # load the stimulus file
        filename = 'locally_light_sparse_noise_72_44_target_size_4_targets_per_frame_5_trials_15_background_-1.0_20190807.npy'#'locally_light_sparse_noise_36_18_target_size_2_targets_per_frame_4_trials_30_background_0.0_20181115.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 2.5 # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        
    elif stimulus_type == 'sd72x44_4':
        # load the stimulus file
        filename = 'locally_dark_sparse_noise_72_44_target_size_4_targets_per_frame_5_trials_15_background_1.0_20190807.npy'#'locally_dark_sparse_noise_36_18_target_size_2_targets_per_frame_4_trials_30_background_-0.0_20181115.npy'
        stimulus['filename'] = filename
        stimulus_tmp = np.load('stimuli/'+filename).item()
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 2.5 # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)

################ end of sparse noised signal ###############################

################ Kerstin/oddball stim   ###############################
    elif stimulus_type == 'Kerstin_short':
        stimulus['trials'] = 10
        stimulus['temporalfrequencies'] = np.array([2.0])#,2.,3.]) # in Hz, maybe here going to 2 would save time? # for LGN: 2,3,4 Hz
        stimulus['spatialfrequencies'] = np.array([0.04]) #np.array([0.01,0.02,0.04,0.08,0.16]) # np.array([0.01,0.02,0.04,0.08,0.16])
        stimulus['size'] =([400.,400.]) # np.array([50.,50.]) #  to change in order to reduce the mask size
        stimulus['phase'] = 0.0
        stimulus['duration'] = 0.5 #NB there 0.5 is to match imaging 0.75 to 1s is recommanded
        stim_duration_backup = 0.5
        stimulus['pause'] = 0.75 # NB poause is used twice
        stimulus['position'] = position
        params['monitor']['background'] = [0.,0.,0.]
        
        analysis['type'] = 'TuningPolar'
        analysis['parameters'] = ['orientations'] 
        analysis['psth_duration'] = stimulus['duration']
        
        stimulus['local_type'] = 'do nothing'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        print "Time passed %d" %(time()-start) + ' seconds'
        stimulus['orientations'] = np.linspace(0.,360.-(360/8.),8) # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        params['stimulus'] = stimulus
        MovingGratingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        
        stimulus['orientations'] = [135,45,45,45,45,45,45,45] # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        params['stimulus'] = stimulus
        MovingGratingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        
        stimulus['orientations'] = [45,45,45,45,45] # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        stimulus['local_type'] = 'repeat'
        params['stimulus'] = stimulus
        MovingGratingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        
        stimulus['duration'] = 10.0
        stimulus['orientations'] = [45] # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        stimulus['local_type'] = 'long'
        params['stimulus'] = stimulus
        MovingGratingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        
        stimulus['local_type'] = 'do nothing'
        stimulus['duration'] = stim_duration_backup
        stimulus['orientations'] = np.linspace(0.,360.-(360/8.),8) # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        params['stimulus'] = stimulus
        MovingGratingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        
        stimulus['orientations'] = [45,135,135,135,135,135,135,135] # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        params['stimulus'] = stimulus
        MovingGratingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        
        stimulus['orientations'] = [135,135,135,135,135] # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        stimulus['local_type'] = 'repeat'
        params['stimulus'] = stimulus
        MovingGratingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        
        stimulus['duration'] = 10.0
        stimulus['orientations'] = [135] # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        stimulus['local_type'] = 'long'
        params['stimulus'] = stimulus
        MovingGratingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        
        stimulus['local_type'] = 'do nothing'
        stimulus['duration'] = stim_duration_backup
        stimulus['orientations'] = np.linspace(0.,360.-(360/8.),8) # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        params['stimulus'] = stimulus
        MovingGratingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        print "Time passed %d" %(time()-start) + ' seconds'

    elif stimulus_type == 'Kerstin_long':
        stimulus['trials'] = 10
        stimulus['temporalfrequencies'] = np.array([2.0])#,2.,3.]) # in Hz, maybe here going to 2 would save time? # for LGN: 2,3,4 Hz
        stimulus['spatialfrequencies'] = np.array([0.04]) #np.array([0.01,0.02,0.04,0.08,0.16]) # np.array([0.01,0.02,0.04,0.08,0.16])
        stimulus['size'] =([400.,400.]) # np.array([50.,50.]) #  to change in order to reduce the mask size
        stimulus['phase'] = 0.0
        stimulus['duration'] = 1.5 #0.5 #NB there 0.5 is to match imaging 0.75 to 1s is recommanded
        stim_duration_backup = 1.5 #0.5
        stimulus['pause'] = 1.25# 0.75 # NB poause is used twice
        stimulus['position'] = position
        params['monitor']['background'] = [0.,0.,0.]
        
        analysis['type'] = 'TuningPolar'
        analysis['parameters'] = ['orientations'] 
        analysis['psth_duration'] = stimulus['duration']
        
        stimulus['local_type'] = 'do nothing'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        print "Time passed %d" %(time()-start) + ' seconds'
        stimulus['orientations'] = np.linspace(0.,360.-(360/8.),8) # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        params['stimulus'] = stimulus
        MovingGratingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        
        stimulus['orientations'] = [135,45,45,45,45,45,45,45] # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        params['stimulus'] = stimulus
        MovingGratingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        
        stimulus['orientations'] = [45,45,45,45,45] # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        stimulus['local_type'] = 'repeat'
        params['stimulus'] = stimulus
        MovingGratingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        
        stimulus['duration'] = 20.0
        stimulus['orientations'] = [45] # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        stimulus['local_type'] = 'long'
        params['stimulus'] = stimulus
        MovingGratingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        
        stimulus['local_type'] = 'do nothing'
        stimulus['duration'] = stim_duration_backup
        stimulus['orientations'] = np.linspace(0.,360.-(360/8.),8) # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        params['stimulus'] = stimulus
        MovingGratingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        
        stimulus['orientations'] = [45,135,135,135,135,135,135,135] # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        params['stimulus'] = stimulus
        MovingGratingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        
        stimulus['orientations'] = [135,135,135,135,135] # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        stimulus['local_type'] = 'repeat'
        params['stimulus'] = stimulus
        MovingGratingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        
        stimulus['duration'] = 20.0
        stimulus['orientations'] = [135] # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        stimulus['local_type'] = 'long'
        params['stimulus'] = stimulus
        MovingGratingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        
        stimulus['local_type'] = 'do nothing'
        stimulus['duration'] = stim_duration_backup
        stimulus['orientations'] = np.linspace(0.,360.-(360/8.),8) # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        params['stimulus'] = stimulus
        MovingGratingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        OurSetup.present_pause(10*120,win)
        print "Time passed %d" %(time()-start) + ' seconds'
    
        
 ################ End Kerstin/oddball stim   ###############################
        
    elif stimulus_type == 'mg':
        stimulus['local_type'] = 'do nothing'
        stimulus['trials'] =10
        stimulus['temporalfrequencies'] = np.array([2.])#,2.,3.]) # in Hz, maybe here going to 2 would save time? # for LGN: 2,3,4 Hz
        stimulus['spatialfrequencies'] = np.array([0.02,0.04,0.08]) #np.array([0.01,0.02,0.04,0.08,0.16]) # np.array([0.01,0.02,0.04,0.08,0.16])
        stimulus['orientations'] = np.linspace(0.,360.-(360/12.),12) # np.array([270.]) #  np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12) #np.linspace(0.,360.-(360/12.),12)  # np.array([210.])
        stimulus['size'] = ([400.,400.]) # np.array([50.,50.]) # ([400.,400.]) # to change in order to reduce the mask size
        stimulus['phase'] = 0.0
        stimulus['duration'] = 1.5
        stimulus['pause'] = 0.25
        stimulus['position'] = position
        params['monitor']['background'] = [0.,0.,0.]
        
        analysis['type'] = 'TuningPolar'
        analysis['parameters'] = ['orientations'] 
        analysis['psth_duration'] = stimulus['duration']
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        MovingGratingStimulus.run_stimulus(params,setup)    
        print "Time passed %d" %(time()-start) + ' seconds'

    elif stimulus_type == 'mb': #mooving bar white or black on gray background
        stimulus['trials'] = 15
        stimulus['orientations'] = np.linspace(0.,360.-(360./12.),12) # np.array([270.])#np.linspace(0.,360.-(360./12.),12)#np.linspace(0.,360.-(360./12.),12)#np.array([120.])
        stimulus['widths'] = np.array([10.])
        stimulus['length'] = 400. # 50. #400
        stimulus['speeds'] = np.array([150.])#np.array([150.]) #degree per seconds
        stimulus['luminances'] = np.array([1.]) # 1 = light, -1 dark, 0 gray  np.array([-1.,1.])
        stimulus['radius_circle'] =  120. #30. #
        params['monitor']['background'] = [-1.,-1.,-1.]
        
        analysis['type'] = 'TuningPolar'
        analysis['parameters'] = ['orientations'] 
        analysis['psth_duration'] = 1.5
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        MovingBarStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
    
    elif stimulus_type == 'mb_thin': #mooving bar white or black on gray background
        stimulus['trials'] = 10
        stimulus['orientations'] = np.linspace(0.,360.-(360./12.),12) # np.array([270.])#np.linspace(0.,360.-(360./12.),12)#np.linspace(0.,360.-(360./12.),12)#np.array([120.])
        stimulus['widths'] = np.array([2.])
        stimulus['length'] = 400. # 50. #400
        stimulus['speeds'] = np.array([150.])#np.array([150.]) #degree per seconds
        stimulus['luminances'] = np.array([1.]) # 1 = light, -1 dark, 0 gray  np.array([-1.,1.])
        stimulus['radius_circle'] =  120. #30. #
        params['monitor']['background'] = [-1.,-1.,-1.]
        
        analysis['type'] = 'TuningPolar'
        analysis['parameters'] = ['orientations'] 
        analysis['psth_duration'] = 1.5
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        MovingBarStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        
    elif stimulus_type == 'mp': #moving point
        stimulus['trials'] = 10
        stimulus['orientations'] = np.linspace(0.,360.-(360./12.),12) #np.array([330.])#np.linspace(0.,360.-(360/12.),12)#np.linspace(0.,360.-(360/12.),12)#np.array([120.])
        stimulus['widths'] = np.array([5.])
        stimulus['length'] = 5.
        stimulus['speeds'] = np.array([150.]) #degree per seconds
        stimulus['luminances'] = np.array([1.]) # 1 = light, -1 dark, 0 gray
        stimulus['radius_circle'] =  90.#  30. #
        params['monitor']['background'] = [-1.,-1.,-1.]
        
        analysis['type'] = 'TuningPolar'
        analysis['parameters'] = ['orientations'] 
        analysis['psth_duration'] = 2.
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        MovingBarStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        
    elif stimulus_type == 'mbg': #mooving bar white or black on gray background
        stimulus['trials'] = 10
        stimulus['offsets'] = np.array([-1.,0.,1.])#np.array([-5,-4,-3.,-2.,-1.,0.,1.,2.,3.,4.,5.]) this multiples to the LENGTH of the bar
        stimulus['orientations'] = np.linspace(0.,360.-(360./12.),12) # np.array([270.])#np.linspace(0.,360.-(360./12.),12)#np.linspace(0.,360.-(360./12.),12)#np.array([120.])
        stimulus['widths'] = np.array([5.])
        stimulus['length'] = 40.#400. # 50. #400
        stimulus['speeds'] = np.array([150.])#np.array([150.]) #degree per seconds
        stimulus['luminances'] = np.array([1.]) # 1 = light, -1 dark, 0 gray  np.array([-1.,1.])
        stimulus['radius_circle'] =  120.#120. #30. #
        params['monitor']['background'] = [-1.,-1.,-1.]#[-1.,-1.,-1.]
        
        analysis['type'] = 'TuningPolar'
        analysis['parameters'] = ['orientations'] 
        analysis['psth_duration'] = 1.5
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        MovingBarGridStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        
    elif stimulus_type == 'cm_36x22': 
        # load the stimulus file
        filename =  'Checkerboard_36_22_n_frames_7920.0_20181120.npy'#'Checkerboard_36_22_n_frames_6480.0__20181114.npy' #'Checkerboard_45_25_target_size_1_n_frames_5000_20181024.npy' 
        
        stimulus['filename'] = filename
        #stimulus_tmp = np.load('stimuli/'+filename)#.item()
        #stimulus_frames = np.load('stimuli/'+filename) #.item()
        stimulus_tmp = {'background':-0.0,'frames':np.load('stimuli/'+filename), 'target': -1.0}
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        # parameters
        params['monitor']['background'] = [0.,0.,0.]
        stimulus['trials'] = 1
        stimulus['scale'] = 5 # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12 #1   # 10 = 80 ms
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STAcm'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        
        
    elif stimulus_type == 'cm_25x25': 
        # load the stimulus file
        filename =  'Checkerboard_25_25_n_frames_5000__20181025.npy'#'Checkerboard_36_22_n_frames_6480.0__20181114.npy' #'Checkerboard_45_25_target_size_1_n_frames_5000_20181024.npy' 
        
        stimulus['filename'] = filename
        #stimulus_tmp = np.load('stimuli/'+filename)#.item()
        #stimulus_frames = np.load('stimuli/'+filename) #.item()
        stimulus_tmp = {'background':-0.0,'frames':np.load('stimuli/'+filename), 'target': -1.0}
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        # parameters
        params['monitor']['background'] = [0.,0.,0.]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12 #1   # 10 = 80 ms
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STAcm'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
    
    elif stimulus_type == 'cm_10x10': 
        # load the stimulus file
        filename =  'Checkerboard_10_10_target_size_1_n_frames_5000__20190430.npy'#'Checkerboard_36_22_n_frames_6480.0__20181114.npy' #'Checkerboard_45_25_target_size_1_n_frames_5000_20181024.npy' 
        
        stimulus['filename'] = filename
        #stimulus_tmp = np.load('stimuli/'+filename)#.item()
        #stimulus_frames = np.load('stimuli/'+filename) #.item()
        stimulus_tmp = {'background':-0.0,'frames':np.load('stimuli/'+filename), 'target': -1.0}
        stimulus_frames = stimulus_tmp['frames']
        background = stimulus_tmp['background']
        
        # parameters
        params['monitor']['background'] = [0.,0.,0.]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12 #1   # 10 = 80 ms
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STAcm'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'

    elif stimulus_type == 'chi':
        # load the stimulus file
        filename = 'chirp_stimulus_20181011.npy'
        stimulus['filename'] = filename
        stimulus_frames = np.load('stimuli/'+filename)
        # parameters
        params['monitor']['background'] = [0.,0.,0.]
        stimulus['trials'] = 20
        stimulus['scale'] = 350. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 1
        stimulus['position'] = position# in deg
        stimulus['duration'] = 35.
                           
        analysis['type'] = 'PSTH' #'TunningLinear'# #'TunningLinear'
        analysis['parameters'] = ['speeds']         
        analysis['psth_duration'] = stimulus['duration']
                
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images_with_first_frame_trigger(stimulus_frames,params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'

    elif stimulus_type == 'chi_fast':
        # load the stimulus file
        filename = 'chirp_stimulus_20181011.npy'
        stimulus['filename'] = filename
        stimulus_frames = np.load('stimuli/'+filename)
        stimulus_frames = stimulus_frames[:,:,0::2]
        # parameters
        params['monitor']['background'] = [0.,0.,0.]
        stimulus['trials'] = 20
        stimulus['scale'] = 350. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 1
        stimulus['position'] = position# in deg
        stimulus['duration'] = 35.
                           
        analysis['type'] = 'PSTH' #'TunningLinear'# #'TunningLinear'
        analysis['parameters'] = ['speeds']         
        analysis['psth_duration'] = stimulus['duration']
                
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images_with_first_frame_trigger(stimulus_frames,params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        
    elif stimulus_type == 'chi_ultra_fast':
        # load the stimulus file
        filename = 'chirp_stimulus_20181011.npy'
        stimulus['filename'] = filename
        stimulus_frames = np.load('stimuli/'+filename)
        stimulus_frames = stimulus_frames[:,:,0::4]
        # parameters
        params['monitor']['background'] = [0.,0.,0.]
        stimulus['trials'] = 20
        stimulus['scale'] = 350. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 1
        stimulus['position'] = position# in deg
        stimulus['duration'] = 35.
                           
        analysis['type'] = 'PSTH' #'TunningLinear'# #'TunningLinear'
        analysis['parameters'] = ['speeds']         
        analysis['psth_duration'] = stimulus['duration']
                
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images_with_first_frame_trigger(stimulus_frames,params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
    
    elif stimulus_type == 'chi_jon_fast':
        # load the stimulus file
        filename = 'ultrafastchirp'#'chirp_stimulus_20181011.npy'
        stimulus['filename'] = filename
        #stimulus_frames = np.load('stimuli/'+filename)
        tframes = 120*30
        tt = np.linspace(0, 10, tframes)
        tw = chirp(tt, f0=0.1, f1=120., t1=10, method='linear')

        stimulus_frames = np.zeros([1,1,tframes])#stimulus_frames[:,:,0::4]
        stimulus_frames[:] = tw
        
        # parameters
        params['monitor']['background'] = [0.,0.,0.]
        stimulus['trials'] = 10
        stimulus['scale'] = 350. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 1
        stimulus['position'] = position# in deg
        stimulus['duration'] = 35.
                           
        analysis['type'] = 'PSTH' #'TunningLinear'# #'TunningLinear'
        analysis['parameters'] = ['speeds']         
        analysis['psth_duration'] = stimulus['duration']
                
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images_with_first_frame_trigger(stimulus_frames,params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        
    elif stimulus_type == 'chi2':
        # load the stimulus file
        filename = 'chirp_stimulus_20190430.npy'
        stimulus['filename'] = filename
        stimulus_frames = np.load('stimuli/'+filename)
        
        # parameters
        params['monitor']['background'] = [0.,0.,0.]
        stimulus['trials'] = 20
        stimulus['scale'] = 350. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 1
        stimulus['position'] = position# in deg
        stimulus['duration'] = 35.
                           
        analysis['type'] = 'PSTH' #'TunningLinear'# #'TunningLinear'
        analysis['parameters'] = ['speeds']         
        analysis['psth_duration'] = stimulus['duration']
                
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images_with_first_frame_trigger(stimulus_frames,params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        
    elif stimulus_type == 'lo':
        params['monitor']['background'] = [1.,1.,1.]
        stimulus['trials'] = 20
        stimulus['speeds'] = np.array([10.,50.,100.,200.])
        stimulus['max_size'] = 100. # this is the size when the disc is at its largest size
        stimulus['luminances'] = np.array([-1.])
        stimulus['position'] = position #np.array([0.,15.]) 
        stimulus['duration'] = 3.
        
        analysis['type'] = 'TunningLinear'# 'PSTH'#'TunningLinear'
        analysis['parameters'] = ['speeds'] 
        analysis['psth_duration'] = stimulus['duration']
                
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        
        LoomingStimulus.run_stimulus(params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
        
    elif stimulus_type == 'testdome':
        # load the stimulus file
        #filename = 'locally_light_sparse_noise_36_22_target_size_3_targets_per_frame_2_trials_10_background_0.0_20181120.npy'#'locally_light_sparse_noise_36_18_target_size_2_targets_per_frame_4_trials_30_background_0.0_20181115.npy
        #stimulus['filename'] = filename
        #stimulus_tmp = np.load('stimuli/'+filename).item()
        #stimulus_frames = stimulus_tmp['frames']        
        
        stimulus_frames = np.ones([10,20,10])*-1.
        stimulus_frames[8,18,:] = 1.
        
        plt.pcolormesh(stimulus_frames[:,:,0])
        
        background = 0.#stimulus_tmp['background']
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 5. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12*200
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(-stimulus_frames,params,setup)
    
    
    elif stimulus_type == 'csd': #
        # load the stimulus fil
        background = 0.
        params['monitor']['background'] = [background,background,background]# [1.,1.,1.]#

        stimulus['trials'] = 200
        stimulus['scale'] = 10. # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 120 # 1 sec
        stimulus['position'] = position# in deg        
#        analysis['type'] = 'STA'        
        params['stimulus'] = stimulus
        params['analysis'] = analysis

        # run
        CSDStimulus.present_images(params,setup)
        
    elif stimulus_type == 'Images':
        stimulus['trials'] = 70 #50
        stimulus['orientations'] =  np.array([0.])
        stimulus['size'] = np.array([250.,250.])  # [250.,250.] for checker to be 10° (2.6cm) at 15 cm distance          300,300 covers full monitor BEWARE APPLIES JUST FOR THE CHECKER
        stimulus['phase'] = 0.0
        stimulus['duration'] = 0.5#1#0.5
        stimulus['pause'] = 0.5 #0.05 #1 #0.5
        stimulus['contrast'] = 1.0 #0.5  #between -1 (negative) and 1 (1.0 = unchanged)
        stimulus['position'] = np.array([0.,0.]) #np.array([-50,-30.])   #position   # np.array([0.,0.]) # negative: left and down, positive: right and up 
        params['monitor']['background'] = [0.,0.,0.]
        
        analysis['type'] = 'PSTH' # 'STA'# #'PSTH'#'PSTH' #'SCCorr'  #'TuningPolar'
        analysis['parameters'] = ['orientations']
        analysis['psth_duration'] = stimulus['duration'] 
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        
        # run
        monitor_type = monitor['type']# condition to load the monitor type variable ('Dell' or 'Dome')  
        NaturalImageStimulusWithGray_CG.run_stimulus(params, setup, monitor_type) # monitore type for dome experiences
        
    elif stimulus_type == 'testpic':
        # load the stimulus file
#        filename = 'locally_dark_sparse_noise_45_25_target_size_3_targets_per_frame_2_trials_10_background_1.0_20181020.npy'
        filename = 'locally_dark_sparse_noise_36_22_target_size_3_targets_per_frame_2_trials_10_background_1.0_20181120.npy'#
        stimulus['filename'] = filename
#        img = plt.imread('C:/Users/AGKremkow/Desktop/Visual-Stimuli-Neuropixel/VisualStimuli/stimuli/square_22x22.png')
        img = plt.imread('C:/Users/AGKremkow/Desktop/Visual-Stimuli-Neuropixel/VisualStimuli/stimuli/Capturecm36-22.png')
        img = img.mean(2)
        img -= img.min()
        img /= img.max()
        img *= 2
        img -= 1.
        
        
#        stimulus_frames = np.zeros([739,739,2])
        stimulus_frames = np.zeros([420,685,2])
        stimulus_frames[:,:,0] = img
        stimulus_frames[:,:,1] = img
        background = 0.
        
        params['monitor']['background'] = [background,background,background]
        stimulus['trials'] = 1
        stimulus['scale'] = 0.2 # scale is now in the units of deg, i.e. scale = 1 means that one noise pixel has the size of 1x1 deg
        stimulus['stimulus_duration_in_frames'] = 12000
        stimulus['position'] = position# in deg    
        
        analysis['type'] = 'STA'
        
        params['stimulus'] = stimulus
        params['analysis'] = analysis
        # run
        ImageStimulus.present_images(stimulus_frames,params,setup)
        print "Time passed %d" %(time()-start) + ' seconds'
     
print "Times past %d" %(time()-start) + ' secondes'
win.close()
# close trigger
trigger.task.close()
## %%