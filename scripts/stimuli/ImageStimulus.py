# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 08:28:24 2018

@author: AGKremkow
"""
# %%
import os
from psychopy import visual
import numpy as np
import time
import our_setup_new as OurSetup

STIMULI_FOLDER = "/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli/Psychopy-36x22" # Psychopy-36x22, Psychopy-64x36-test
# WIDTH = 64 # 2560  # 2560
# HEIGHT = 36 # 1440  # 1440

WIDTH = 36
HEIGHT = 22

# %%
def present_images(stimulus_frames,params,setup):
     # %% we create some filenames, the tmp is used for the communication with the stimulus pc. the save for saving ... haha 
    # filename_save = OurSetup.generate_filename_and_make_folders(params)
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
    print("stim_size: ", stim_size)
    print("scale: ", scale)
    print("size: ", stim_size*scale)
    for frameN in range(n_frames):
        # dome fliplr and flipud
        if params['monitor']['type'] == 'Dome':
#            print(stim_size*scale)
            images[str(frameN)]=visual.ImageStim(win,np.flipud(np.fliplr(stimulus_frames[:,:,frameN])),size=stim_size*scale,pos=position) #needed for dome setup # JK 20190424
#            images[str(frameN)]=visual.ImageStim(win,stimulus_frames[:,:,frameN],size=stim_size*scale,pos=position) #needed for dome setup # JK 20190424
        else:
            images[str(frameN)] = visual.ImageStim(
                win,
                stimulus_frames[:,:,frameN],
                size=stim_size*scale,
                pos=position,
                # units='deg',
                # interpolate=False
            )
   # %%
    print("Generating numpy matrices for Stimulus: ", params['stimulus']['type'])
    # we save the params
    # if not params['test']:
        # we save the params in the folder, always needed. Just in test maybe not
        # np.save(filename_save,params)
        # we wait for the online analysis
        # if params['closed_loop_with_analysis']:
            # OurSetup.write_current_params_and_wait_for_go(params)
            
    for trial in range(params['stimulus']['trials']):
        # we present some text outside the screen to get the system up and running
        OurSetup.present_pause(120,win)# in frames
        # present images

        # psychopy_frames = []
        psychopy_frames = np.empty(
            (n_frames, HEIGHT, WIDTH),
            dtype=np.uint8
        )
        for frameN in range(n_frames):
            # psychopy_sub_frames = []
            for flipN in range(stimulus_duration_in_frames):
                images[str(frameN)].draw()
                win.flip()
                
                # Append only one frame to be saved (cause that's when the TTL was triggered)
                if flipN == 0:
                    frame = win.getMovieFrame(buffer='front')
                    frame_np = np.array(frame.convert('L'))
                    vertically_flipped_frame = np.flip(frame_np, 0)
                    # psychopy_frames.append(vertically_flipped_frame)
                    psychopy_frames[frameN] = vertically_flipped_frame
                    # trigger.FrameTime()

            # if len(psychopy_frames) > 0:
            #     psychopy_frames = np.concatenate((psychopy_frames, np.array(psychopy_sub_frames)), axis=0)
            # else:
            #     psychopy_frames = np.array(psychopy_sub_frames)
        
        # Save Psychopy frames
        # psychopy_frames = np.array(psychopy_frames)
        print(psychopy_frames.shape)
        
        if params['stimulus']['type'] == 'cm_36x22':
            file_name = f'checkermap_{WIDTH}_{HEIGHT}.npy'
        elif params['stimulus']['type'] == 'sd36x22_3':
            file_name = f'sparse_noise_dark_{WIDTH}_{HEIGHT}.npy'
        elif params['stimulus']['type'] == 'sl36x22_3':
            file_name = f'sparse_noise_light_{WIDTH}_{HEIGHT}.npy'
        elif params['stimulus']['type'] == 'NaturalMovie':
            file_name = f'natural_movie_{WIDTH}_{HEIGHT}.npy'
        elif params['stimulus']['type'] == 'PhaseScrblMovie':
            file_name = f'natural_movie_scrable_{WIDTH}_{HEIGHT}.npy'
        elif params['stimulus']['type'] == 'SwapMovie':
            file_name = f'natural_movie_swap_{WIDTH}_{HEIGHT}.npy'

        file_path = os.path.join(STIMULI_FOLDER, file_name)
        with open(file_path, 'wb') as f:
            np.save(f, psychopy_frames)
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
    win = OurSetup.OpenScreen(background_color, params['monitor']['distance'], params['monitor']['type'])
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

    print("n_frames: ", n_frames)
    print("stimulus_duration_in_frames", stimulus_duration_in_frames)
    
    for frameN in range(n_frames):
        images[str(frameN)]=visual.ImageStim(win,stimulus_frames[:,:,frameN],size=stim_size*scale,pos=position)
    # %%
    
    # we save the params
    if not params['test']:
        # we save the params in the folder, always needed. Just in test maybe not
        # np.save(filename_save,params)
        # we wait for the online analysis
        if params['closed_loop_with_analysis']:
            OurSetup.write_current_params_and_wait_for_go(params)
            
    if params['stimulus']['type'] == 'chi':
        file_name = f'chirp_{WIDTH}_{HEIGHT}.npy'

    psychopy_frames = np.empty(
        (n_frames*params['stimulus']['trials'], HEIGHT, WIDTH),
        dtype=np.uint8
    )
    for trial in range(params['stimulus']['trials']):
        # we present some text outside the screen to get the system up and running
        OurSetup.present_pause(120,win)# in frames
        
        # present images
        for frameN in range(n_frames):
            for flipN in range(stimulus_duration_in_frames):
                images[str(frameN)].draw()
                win.flip()
                if frameN == 0:
                    frame = win.getMovieFrame(buffer='front')
                    frame_np = np.array(frame.convert('L'))
                    vertically_flipped_frame = np.flip(frame_np, 0)
                    psychopy_frames[frameN] = vertically_flipped_frame
                    # trigger.FrameTime()
                    triggercount += 1
                    # print(str(triggercount)+'-'+str(params['stimulus']['trials']))
        
        file_path = os.path.join(STIMULI_FOLDER, file_name)
        with open(file_path, 'wb') as f:
            np.save(f, psychopy_frames)
        # we flip one more time to make the screen gray
        win.flip()
        OurSetup.present_pause(360,win)# in frames
    
    # %%
    #OurSetup.present_pause(240,win)# in frames
    
    # trigger.Stimulus_Stop()
