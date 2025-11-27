# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 08:28:24 2018

@author: AGKremkow
"""
# %%
from psychopy import visual
import numpy as np
import time
import our_setup_new as OurSetup

# %%
def present_images(stimulus_frames,params,setup):
     # %% we create some filenames, the tmp is used for the communication with the stimulus pc. the save for saving ... haha 
    filename_save = OurSetup.generate_filename_and_make_folders(params)
    # %% Create window
    win = setup['win']
    # trigger = setup['trigger']
    background_color = params['monitor']['background']
    background_color
    win.setRGB(background_color)
    win.flip()
    win = OurSetup.OpenScreen(background_color, params['monitor']['distance'], params['monitor']['type'])
    # trigger.Stimulus_Start()
     # we present background for few second such that the retina can adapt
    onset_adaptation_time_sec = params['stimulus']['onset_adaptation_time_sec']
    onset_adaptation_time_frames = int(np.round(onset_adaptation_time_sec * 120))
    OurSetup.present_pause(onset_adaptation_time_frames,win)
    # %% generate images
    scale = params['stimulus']['scale']
    position = params['stimulus']['position']
    position = position*-1. # because ofthe funcky flipping of the axis
    stimulus_duration_in_frames = params['stimulus']['stimulus_duration_in_frames']
    n_frames = stimulus_frames.shape[2]
    stim_size = np.array([stimulus_frames.shape[1],stimulus_frames.shape[0]]).astype('float')
    images = {}
        
    for frameN in range(n_frames):
        # dome fliplr and flipud
        if params['monitor']['type'] == 'Dome':
#            print(stim_size*scale)
            images[str(frameN)]=visual.ImageStim(win,np.flipud(np.fliplr(stimulus_frames[:,:,frameN])),size=stim_size*scale,pos=position) #needed for dome setup # JK 20190424
#            images[str(frameN)]=visual.ImageStim(win,stimulus_frames[:,:,frameN],size=stim_size*scale,pos=position) #needed for dome setup # JK 20190424
        else:
            images[str(frameN)]=visual.ImageStim(win,stimulus_frames[:,:,frameN],size=stim_size*scale,pos=position)
   # %%
    
    # we save the params
    if not params['test']:
        # we save the params in the folder, always needed. Just in test maybe not
        np.save(filename_save,params)
        # we wait for the online analysis
        if params['closed_loop_with_analysis']:
            OurSetup.write_current_params_and_wait_for_go(params)
            
    for trial in range(params['stimulus']['trials']):
        # we present some text outside the screen to get the system up and running
        OurSetup.present_pause(120,win)# in frames
        # present images
        for frameN in range(n_frames):
            for flipN in range(stimulus_duration_in_frames):
                images[str(frameN)].draw()
                win.flip()
                #raw_input()
                # if flipN == 0:
                    # trigger.FrameTime()
        # we flip one more time to make the screen gray
        win.flip()
        OurSetup.present_pause(360,win)# in frames
    
    # %%
    #OurSetup.present_pause(240,win)# in frames
    
    # trigger.Stimulus_Stop()
    #win.close()


### NB this section is solely used for the chirp i.e. no oreintation what so ever
def present_images_with_first_frame_trigger(stimulus_frames,params,setup):
     # %% we create some filenames, the tmp is used for the communication with the stimulus pc. the save for saving ... haha 
    filename_save = OurSetup.generate_filename_and_make_folders(params)
    # %% Create window
    win = setup['win']
    # trigger = setup['trigger']
    background_color = params['monitor']['background']
    win.setRGB(background_color)
    win.flip()
    win = OurSetup.OpenScreen(background_color,monitor_distance)
    # trigger.Stimulus_Start()
     # we present background for few second such that the retina can adapt
    onset_adaptation_time_sec = params['stimulus']['onset_adaptation_time_sec']
    onset_adaptation_time_frames = int(np.round(onset_adaptation_time_sec * 120))
    OurSetup.present_pause(onset_adaptation_time_frames,win)
    # %% generate images
    scale = params['stimulus']['scale']
    position = params['stimulus']['position']
    stimulus_duration_in_frames = params['stimulus']['stimulus_duration_in_frames']
    n_frames = stimulus_frames.shape[2]
    stim_size = np.array([stimulus_frames.shape[1],stimulus_frames.shape[0]]).astype('float')
    triggercount = 0
    n_stimuli = n_frames
    images = {}
    
    for frameN in range(n_frames):
        images[str(frameN)]=visual.ImageStim(win,stimulus_frames[:,:,frameN],size=stim_size*scale,pos=position)
    # %%
    
    # we save the params
    if not params['test']:
        # we save the params in the folder, always needed. Just in test maybe not
        np.save(filename_save,params)
        # we wait for the online analysis
        if params['closed_loop_with_analysis']:
            OurSetup.write_current_params_and_wait_for_go(params)
            
    
    for trial in range(params['stimulus']['trials']):
        # we present some text outside the screen to get the system up and running
        OurSetup.present_pause(120,win)# in frames
        # present images
        for frameN in range(n_frames):
            for flipN in range(stimulus_duration_in_frames):
                images[str(frameN)].draw()
                win.flip()
                if frameN == 0:
                    # trigger.FrameTime()
                    triggercount += 1
                    print(str(triggercount)+'-'+str(params['stimulus']['trials']))
        # we flip one more time to make the screen gray
        win.flip()
        OurSetup.present_pause(360,win)# in frames
    
    # %%
    #OurSetup.present_pause(240,win)# in frames
    
    # trigger.Stimulus_Stop()
