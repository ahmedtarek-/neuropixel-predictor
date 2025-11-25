# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 07:17:58 2018

@author: AGKremkow
"""

# this code presents a moving bar
# %%
import matplotlib.pyplot as plt
import numpy as np
# import scipy.io as io
from psychopy import visual, monitors
import OurSetup
# import time

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
     
    refreshrate = params['monitor']['refreshrate']
    onset_adaptation_time_sec = params['stimulus']['onset_adaptation_time_sec']
    onset_adaptation_time_frames = int(np.round(onset_adaptation_time_sec * refreshrate))
    OurSetup.present_pause(onset_adaptation_time_frames,win)# in frames
    
    # %%
    
    bar_widths_deg = params['stimulus']['widths']
    bar_length_deg = params['stimulus']['length']
    bar_orientations_deg = params['stimulus']['orientations']
    bar_speeds_deg = params['stimulus']['speeds']
    bar_luminances = params['stimulus']['luminances']
    
    radius_circle_deg = params['stimulus']['radius_circle'] # now in deg, this will be the circle, each bar will move from one side to the other
    
    
    n_trials = params['stimulus']['trials']
    
    
    
    # %%
    all_orientations_in_deg = []
    all_speeds_in_deg = []
    all_widths_in_deg = []
    all_luminances = []
    
    
    for ori in bar_orientations_deg:
        for speed in bar_speeds_deg:
            for width in bar_widths_deg:
                for lum in bar_luminances:
                    all_orientations_in_deg.append(ori)
                    all_speeds_in_deg.append(speed)
                    all_widths_in_deg.append(width)
                    all_luminances.append(lum)
    
    all_orientations_in_deg = np.array(all_orientations_in_deg)
    all_speeds_in_deg = np.array(all_speeds_in_deg)
    all_widths_in_deg = np.array(all_widths_in_deg)
    all_luminances = np.array(all_luminances)
    
    # %we randomize the sequenze
    np.random.seed(1675) # this is to have always the same random sequence. However, we need to save the sequence to a file because when we change the parameters the sequences changes
    
    all_orientations_in_deg_all_trials = []
    all_speeds_in_deg_all_trials = []
    all_widths_in_deg_all_trials = []
    all_luminances_all_trials = []
    
    for t in range(n_trials):
        rand_ind = np.random.permutation(len(all_orientations_in_deg))
        # following line to linearized the orientations
        #rand_ind = np.array(range(len(all_orientations_in_deg))) # JK 20190424
        tmp_orientations = all_orientations_in_deg[rand_ind]
        tmp_speeds = all_speeds_in_deg[rand_ind]
        tmp_widths = all_widths_in_deg[rand_ind]
        tmp_lums = all_luminances[rand_ind]
    
        all_orientations_in_deg_all_trials = np.concatenate((all_orientations_in_deg_all_trials,tmp_orientations),axis=0)
        all_speeds_in_deg_all_trials = np.concatenate((all_speeds_in_deg_all_trials,tmp_speeds),axis=0)
        all_widths_in_deg_all_trials = np.concatenate((all_widths_in_deg_all_trials,tmp_widths),axis=0)
        all_luminances_all_trials = np.concatenate((all_luminances_all_trials,tmp_lums),axis=0)
    
    
    sequence = {}
    sequence['orientations'] = all_orientations_in_deg_all_trials
    sequence['speeds'] = all_speeds_in_deg_all_trials
    sequence['widths'] = all_widths_in_deg_all_trials
    sequence['luminances'] = all_luminances_all_trials
    params['stimulus']['sequence'] = sequence
    
    #params['stimulus']['orientation_sequence'] = all_orientations_in_deg_all_trials
    #params['stimulus']['speed_sequence'] = all_speeds_in_deg_all_trials
    #params['stimulus']['width_sequence'] = all_widths_in_deg_all_trials
    #params['stimulus']['luminance_sequence'] = all_luminances_all_trials
    
    # we save the params
    if not params['test']:
        # we save the params in the folder, always needed. Just in test maybe not
        np.save(filename_save,params)
        # we wait for the online analysis
        if params['closed_loop_with_analysis']:
            OurSetup.write_current_params_and_wait_for_go(params)

    
    # %% looping to plot the stimulis    
    triggercount = 0
    n_stimuli = len(all_orientations_in_deg_all_trials)
    
    
    # now we scan the parameter space
    for i in range(n_stimuli):
        # current parameters
        tmp_ori_deg = all_orientations_in_deg_all_trials[i]
        #tmp_ori_deg_remap = 360.- tmp_ori_deg # this step is needed for the polar plot default 
        tmp_ori_deg_remap = 360.-tmp_ori_deg # this step is needed for the polar plot default 
        
        
        # dome start 
        if params['monitor']['type'] == 'Dome':
            if tmp_ori_deg_remap > 180.:
                tmp_ori_deg_remap = tmp_ori_deg_remap - 0.
            else:          
                tmp_ori_deg_remap = tmp_ori_deg_remap - 360.
         
            tmp_ori_deg_remap = 180. + tmp_ori_deg_remap
        # dome end
        
        print tmp_ori_deg        
        
        tmp_ori_rad_remap = tmp_ori_deg_remap/360.*np.pi*2 # we need it in rad 
        
        tmp_speed_deg_sec = all_speeds_in_deg_all_trials[i] # deg/sec
        tmp_speed_deg_frame = tmp_speed_deg_sec/refreshrate
        tmp_width = all_widths_in_deg_all_trials[i]
        tmp_lum = all_luminances_all_trials[i]
        
        # start position
        local_center_x_deg = np.cos(tmp_ori_rad_remap)*radius_circle_deg
        local_center_y_deg = - np.sin(tmp_ori_rad_remap)*radius_circle_deg
        
        local_center_x_deg = local_center_x_deg *-1 # this is for the remapping to the polar plot defaults
        local_center_y_deg = local_center_y_deg *-1
            
        ## the following line is drawing a red circle illustrating startijng and ending points
        line=visual.ImageStim(win,np.ones((1,1))*tmp_lum,size=(tmp_width,bar_length_deg),ori=tmp_ori_deg_remap,pos=(local_center_x_deg,local_center_y_deg),units='deg')
        #line.draw()
        #stim=visual.TextStim(win,text=str(tmp_ori_deg),color=(1.0, 0.0, 0.0),bold=True,height=30.)# JK
        
        if params['test']:  
            point1=visual.ImageStim(win,np.ones((1,1))*-1.,size=(0.1,0.1),ori=0.,pos=(local_center_x_deg,local_center_y_deg),units='deg')
            point=visual.ImageStim(win,np.ones((1,1))*-1.,size=(0.1,0.1),ori=0.,pos=(0.,0.),units='deg')
            ## the following line is drawing a red circle illustrating starting and ending points
#            circle = visual.Circle(win, radius=radius_circle_deg, edges=100, lineColor=[1.,-1.,-1.])
        
        diameter_start_circle_deg = 2.*radius_circle_deg
        duration_in_frames = int(np.round(diameter_start_circle_deg/tmp_speed_deg_frame))
            
        OurSetup.present_pause(80,win)
        
        for frameN in range(duration_in_frames):
            dx = (-np.cos(tmp_ori_rad_remap)*-1.)*tmp_speed_deg_frame
            dy = (np.sin(tmp_ori_rad_remap)*-1.)*tmp_speed_deg_frame
            
            
#            stim.draw() # JK
            
            line.pos += (dx,dy)
            line.draw()
            
            
                       
            if params['test']:  
                point1.pos += (dx,dy)
                point1.draw()
                point.draw()
                #circle.draw()
            
            win.flip()
                
            # you have to send the trigger after a monitor flip!!!
            if frameN == 0:
                trigger.FrameTime()
                triggercount += 1
                print(str(triggercount)+'-'+str(n_stimuli))
        # done 
        pause_duration_frames = int(np.round(0.5*refreshrate))
        OurSetup.present_pause(pause_duration_frames,win)
    
    # %%
    pause_duration_frames = int(np.round(1.*refreshrate))
    OurSetup.present_pause(pause_duration_frames,win)
    trigger.Stimulus_Stop()
    #win.close()