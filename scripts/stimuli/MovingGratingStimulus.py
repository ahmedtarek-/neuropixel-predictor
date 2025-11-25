# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 12:41:06 2018

@author: AGKremkow
"""
  # http://www.djmannion.net/psych_programming/vision/draw_gratings/draw_gratings.html
  
  # To DO: add direction and temporal frequency to the parameter space
  # sf in cpd!
  # temporal frequency in Hz!
  # add scaling factor
  # direction movement raus weil orientation da ()
# %%
from psychopy import visual
import matplotlib.pyplot as plt 
import numpy as np
import OurSetup


# %%
def run_stimulus(params,setup):
    # %% we create some filenames, the tmp is used for the communication with the stimulus pc. the save for saving ... haha 
    filename_save = OurSetup.generate_filename_and_make_folders(params)
    # %% Create window
    win = setup['win']
    trigger = setup['trigger']
    background_color = params['monitor']['background']
    win.setRGB(background_color)
    win.flip()
    #win,trigger = OurSetup.OpenScreen(background_color,monitor_distance)
    trigger.Stimulus_Start()
     # we present background for few second such that the retina can adapt
    onset_adaptation_time_sec = params['stimulus']['onset_adaptation_time_sec']
    onset_adaptation_time_frames = int(np.round(onset_adaptation_time_sec * 120))
    OurSetup.present_pause(onset_adaptation_time_frames,win)
    # %% parameters
    
    position = params['stimulus']['position']
    position = position*-1. # because ofthe funcky flipping of the axis
    
    grating_size = params['stimulus']['size']
    phase_init = params['stimulus']['phase']
    
    orientations = params['stimulus']['orientations']
    spatialfrequencies = params['stimulus']['spatialfrequencies']
    temporalfrequencies = params['stimulus']['temporalfrequencies']
     
    all_oris = []
    all_sfs = []
    all_tfs = []
       
    for ori in orientations:
        for sf in spatialfrequencies:
            for tf in temporalfrequencies:
                    all_oris.append(ori)
                    all_sfs.append(sf)
                    all_tfs.append(tf)
                    
    all_oris = np.array(all_oris)
    all_sfs = np.array(all_sfs)
    all_tfs = np.array(all_tfs)
      
    # % we randomize the sequenze
    np.random.seed(1675) # this is to have always the same random sequence. However, we need to save the sequence to a file because when we change the parameters the sequences changes
    
    n_trials = params['stimulus']['trials']
    
    orientation_sequence = []
    spatialfrequency_sequence = []
    temporalfrequency_sequence = []
    
    
    for t in range(n_trials):
        rand_ind = np.random.permutation(len(all_oris))
        
        tmp_orientations = all_oris[rand_ind]
        tmp_sfs = all_sfs[rand_ind]
        tmp_tfs = all_tfs[rand_ind]
    
        orientation_sequence = np.concatenate((orientation_sequence,tmp_orientations),axis=0)
        spatialfrequency_sequence = np.concatenate((spatialfrequency_sequence,tmp_sfs),axis=0)
        temporalfrequency_sequence = np.concatenate((temporalfrequency_sequence,tmp_tfs),axis=0)
        
        
    #params = {}
    sequence = {}
    sequence['orientations'] = orientation_sequence
    sequence['spatialfrequencies'] = spatialfrequency_sequence
    sequence['temporalfrequencies'] = temporalfrequency_sequence
    params['stimulus']['sequence'] = sequence
    #params['stimulus']['orientation_sequence'] = orientation_sequence
    #params['stimulus']['spatialfrequency_sequence'] = spatialfrequency_sequence
    #params['stimulus']['temporalfrequency_sequence'] = temporalfrequency_sequence
    
    # we save the params
    if not params['test']:
        # we save the params in the folder, always needed. Just in test maybe not
        np.save(filename_save,params)
        # we wait for the online analysis
        if params['closed_loop_with_analysis']:
            OurSetup.write_current_params_and_wait_for_go(params)
    
    # %%
    
    refreshrate = params['monitor']['refreshrate']
    
    #tf = 1.# 2. # in Hz, hence we have to translate tf into how many phase steps we need to complete one cycle in one sec.
    #d_phase = tf/refreshrate
    
    stimulus_duration_sec = params['stimulus']['duration']
    stimulus_duration_frames = np.round(stimulus_duration_sec*refreshrate).astype('int')
    pause_duration_sec = params['stimulus']['pause']
    pause_duration_frames = np.round(pause_duration_sec*refreshrate).astype('int')
    #trial_duration = stimulus_duration_sec + (2*pause_duration_sec)

    # %% Create a grating stimulus
    triggercount = 0      
    choice_format = params['header']['name']
    if choice_format == 'V1_Kerstin':
        grating = visual.GratingStim(win=win, pos=position, size=grating_size, sf=spatialfrequency_sequence[0], ori=360.-orientation_sequence[0],phase=0.,units='deg',tex='sqr') # tex='sqr' to present square gratings instead of sinusoidal
    else:
        grating = visual.GratingStim(win=win, pos=position, size=grating_size, sf=spatialfrequency_sequence[0], ori=360.-orientation_sequence[0],phase=0.,units='deg') #, mask='circle')  
#   
#    OurSetup.present_pause(80,win)
#    index_where_to_make_a_pause = np.array([4,9,14,19,24,29,34,39,44,49]) # this if for the Kersting V1 oddbal paradigm control repeat protocol
#    stimulus_changes_variable = params['stimulus']['local_type']
    
    for ind in range(len(orientation_sequence)):
        # we have to generate the grating
        ori_tmp = 360.-orientation_sequence[ind] # we have to remapp the orientation such that it fits with the default way of polar plot
        print orientation_sequence[ind]
        
        #####  dome start #####
        if params['monitor']['type'] == 'Dome':
            if ori_tmp > 180.:
                ori_tmp = ori_tmp - 0.
            else:          
                ori_tmp = ori_tmp - 360.
            ori_tmp = 180. + ori_tmp
            ###### dome end ######       
        
        sf_tmp = spatialfrequency_sequence[ind]
        tf_tmp = temporalfrequency_sequence[ind]
        
        d_phase = tf_tmp/refreshrate 
        
        grating = visual.GratingStim(win=win, pos=position, size=grating_size, sf=sf_tmp, ori=ori_tmp,phase=phase_init,units='deg')    
        
        # first we need to present the gray period before the grating
        OurSetup.present_pause(pause_duration_frames,win)
        #  Move the stimulus: Create loop where you change spatial phase, orientation or position
        for frameN in range(stimulus_duration_frames):     # 1 frame = 8ms, range = duration of 1 frame/image  , 500 / 120 =>500 = 4 sec, 200/120-->1.6s
            #% draw stimulus and update window
            grating.setPhase(d_phase, '+')    #, '+')            # spatial phase = t*n[Hz], value between 0 and 1 (*n Hz): 0.25 shifts image by half a cycle advance phase by X of a cycle. temporal frequency has modulus 1 (speed > faster), direction of moving: '+' or '-'
            #grating.setPos()                       # set orientation 0-360 in deg
            grating.draw()
            #win.update()
            win.flip()
            if frameN == 0:
                trigger.FrameTime()        # only at stim onset
                triggercount += 1
                print(str(triggercount)+'-'+str(len(orientation_sequence)))
        # % pause after the stimulus
#        OurSetup.present_pause(pause_duration_frames,win)
    # %% cleanup
#        if np.isin(ind,index_where_to_make_a_pause):
#            if stimulus_changes_variable == 'repeat':
#                local_long_pause = np.round(5*refreshrate).astype('int')
#                OurSetup.present_pause(local_long_pause,win)# extra 5s pause
#                print 'hello kitty'
#            elif stimulus_changes_variable == 'long':
#                local_long_pause = np.round(5*refreshrate).astype('int')
#                OurSetup.present_pause(local_long_pause,win)# extra 5s pause
#                print 'hello kitty'
#            else:
#                OurSetup.present_pause(120,win)# normal standart pause
      
    trigger.Stimulus_Stop()          
   # win.close()
