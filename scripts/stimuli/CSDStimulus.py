# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 08:28:24 2018

@author: AGKremkow
"""
# %%
from psychopy import visual
import numpy as np
import time



from psychopy import core
# from ctypes import windll       # load parallel port library
# from psychopy.hardware import crs
from psychopy import visual
from psychopy import monitors
import time
import os
import numpy as np
from psychopy.visual.windowwarp import Warper
# import nidaqmx      # do not use port0.0, port0.1=4, 2=8, 3=16,...

def get_default_parameters():
    header = {}
    header['folder_tmp'] = u'\\\\ONLINEANALYSIS\\exchange-folder\\onlinetmp\\'
    header['folder_save'] = u'\\\\ONLINEANALYSIS\\exchange-folder\\onlinedata\\'
    
    monitor = {}
    monitor['background'] = [0.,0.,0.]
    monitor['refreshrate'] = 120.
    
    # stimulus params
    stimulus = {}
    stimulus['onset_adaptation_time_sec'] = 2.

    # analysis
    analysis = {}
    
    

# # %%
# class Trigger:
#     def __init__(self):
# #        self.portadress = int('0x0378', 16) 
# #        #self.portadress = int('0x0278', 16) 
# #        self.port = windll.inpoutx64
# #        self.port.Out32(self.portadress,0)
#         task = nidaqmx.Task()
#         task.do_channels.add_do_chan('Dev1/port0/line0:7')
#         task.start()
#         self.task = task
        
#     def triggger(self,dataout):
# #        duration = 0.002
# #        self.port.Out32(self.portadress,dataout)
# #        core.wait(duration)
# #        self.port.Out32(self.portadress,0)
        
#         #duration = 0.00001#0.001 #0.1
#         #for i in range(10):
#         self.task.write([dataout])
#         self.task.write([0])
        
#         #self.port.Out32(self.portadress,dataout)
#         # core.wait(duration)
#         #time.sleep(duration)
#         #self.port.Out32(self.portadress,0)
        
#     def FrameTime(self):
#         #  self.triggger(32)  # before seewiesen
#         self.triggger(2) # pin 2 is 2 in dec 010000 (WARNING BINARY CODE HERE)
        
#     def Stimulus_Start(self):
# #        self.triggger(1) # pin 1 is 1 dec 100000 (WARNING BINARY CODE HERE)
#         self.triggger(4) #(4) # pin 1 is 1 dec 100000 (WARNING BINARY CODE HERE)    
        
#         # now we write the params to the folder TO BE UNCOMMENT
# #        folder_save = params['header']['folder_started_protocols']
# #        protocol_counter = params['header']['protocol_counter']
# #        date = params['header']['date']
# #        penetration = params['header']['penetration']
# #        # label = params['header']['label']
# #        stimulus = params['stimulus']['type']
# #        filname_tmp = date+'_'+str(protocol_counter).zfill(3)+'_'+penetration+'_'+stimulus+'_params.npy'
# #        filename_save = folder_save+'\\'+filname_tmp
# #        np.save(filename_save,params)
#         # TO BE UNCOMMENT
        
#         #  send params name to openephys
# #        with zmq.Context() as ctx:
# #            with ctx.socket(zmq.REQ) as sock:
# #                sock.connect('tcp://%s:%d' % (hostname, port))
# #                try:
# #                    # req = raw_input('> ')
# #                    sock.send_string(filname_tmp)
# #                    rep = sock.recv_string()
# #                    print(rep)    
# #                except EOFError:
# #                    print()  # Add final newline
        
#     def Stimulus_Stop(self):
# #        self.triggger(4) # pin 3 is 4 in dec 001000 (WARNING BINARY CODE HERE)
#         self.triggger(16) # pin 3 is 4 in dec 001000 (WARNING BINARY CODE HERE)
#         # now we write the params to the folder TO BE UNCOMMENT
# #        folder_save = params['header']['folder_finished_protocols']
# #        protocol_counter = params['header']['protocol_counter']
# #        date = params['header']['date']
# #        penetration = params['header']['penetration']
# #        #label = params['header']['label']
# #        stimulus = params['stimulus']['type']
# #        filename_save = folder_save+'\\'+date+'_'+str(protocol_counter).zfill(3)+'_'+penetration+'_'+stimulus+'_params.npy'
# #        np.save(filename_save,params)
# #    
# #    def triggger_protocol(self):
# #        duration_local = 0.002
# #        self.port.Out32(self.portadress,8)
# #        core.wait(duration_local)
# #        self.port.Out32(self.portadress,0)
#         #TO BE UNCOMMENT
        
        
def OpenScreen(background_color,monitor_distance,monitor_type):
    ## the fopllowing paragraph is to load the dell monitor high resolution
    #DellMonitor = monitors.Monitor('Dell_B50_C60_good_cal_sept_18')
    #DellMonitor = monitors.Monitor('Dell_B70_C70_good_cal_20180928') # Jens did this. Blame him!
    #DellMonitor = monitors.Monitor('DellCARO20181128bis') # Jens did this. Blame him!
    #DellMonitor.setDistance(monitor_distance)
#    win= visual.Window(monitor = DellMonitor,screen=0,waitBlanking=True,size=[2560, 1440],fullscr=True,allowGUI=False,units='deg',color=background_color, useFBO=True)
    
    ## the following paragraph is to load the NEC projector Dome set-up in november
    #DellMonitor = monitors.Monitor('NEC-HDMI-Jerem-2018-11-08') # Jerem did this..... just trying you know!
    # warper version unknow....
    if monitor_type == 'Dome':
        ## functionnal paragraph for the dome set-up in february
        background_color = [-1,-1,-1]
#        DellMonitor = monitors.Monitor('NEC_20190201good') #'NEC_20190201good' NEC is the projector for the dome
#        DellMonitor = monitors.Monitor('NEC_20190709good') #'NEC_20190201good' NEC is the projector for the dome
#        DellMonitor = monitors.Monitor('NEC_20190712good') #'NEC_20190201good' NEC is the projector for the dome
#        DellMonitor = monitors.Monitor('NEC_20191202good') #'NEC_20190201good' NEC is the projector for the dome
        DellMonitor = monitors.Monitor('NEC_20191202good_Seewiesen20200106')# NEC_20191202good_Seewiesen20200106 with Caro 
        
        DellMonitor.setDistance(monitor_distance)
        win= visual.Window(monitor = DellMonitor,screen=0,waitBlanking=True,size=[1280, 800],fullscr=True,allowGUI=False,units='deg',color=background_color, useFBO=True)
        ## the following line is the used warping for the dome exp.
        #warper = Warper(win, warp= 'warpfile', warpfile = 'test_xyuv.data', eyepoint = [0.5, 0.5], flipHorizontal = False, flipVertical = False) # older
        #warper2 = Warper(win, warp= 'warpfile', warpfile = 'test_xyuv_flat.data', eyepoint = [0.5, 0.5], flipHorizontal = False, flipVertical = False) #older
#        warper14 = Warper(win, warp= 'spherical', eyepoint = [0.5, 0.5],  flipHorizontal =False, flipVertical =False)
#        warper14 = Warper(win, warp= 'spherical', eyepoint = [0.5, 0.35],  flipHorizontal =False, flipVertical =False)
#        warper14 = Warper(win, warp= 'warpfile', warpfile = 'test_xyuv_final_late.data', eyepoint = [0.5, 0.5],  flipHorizontal =False, flipVertical =False) # good version from the 11th july 2019
        warper14 = Warper(win, warp= 'warpfile', warpfile = 'test_xyuv_20190712_early_js.data', eyepoint = [0.5, 0.5],  flipHorizontal =False, flipVertical =False) # good version from the 12th july 2019
#        warper14 = Warper(win, warp= 'warpfile', warpfile = 'test_xyuv_super_weird.data', eyepoint = [0.5, 0.5],  flipHorizontal =False, flipVertical =False)
        
        ## end of the dome paragraph
        print('dome options selected')
    else:
        ## functionnal paragraph for the Dell screen in the neuropixel set-up
        background_color = [-1,-1,-1]
        DellMonitor = monitors.Monitor('Dell_20190626') # to be updated
        DellMonitor.setDistance(monitor_distance)
        win= visual.Window(monitor = DellMonitor,screen=0,waitBlanking=True,size=[2560, 1440],fullscr=True,allowGUI=False,units='deg',color=background_color)
        print('dell screen options selected')
        
          
    trigger = Trigger()
    time.sleep(1.)
    
    return win,trigger


def present_pause(n_frames,win,trigger=None):
    stim=visual.TextStim(win)
    for i in range(n_frames):
        stim.text = str(n_frames-i)
        stim.pos = [-3000,0]
        stim.draw()
        win.flip()
#        if trigger is not None:
#            trigger.EyeCamera()
        
def write_current_params_and_wait_for_go(params):
    tmp_dir = params['header']['folder_tmp']
    filename_tmp = tmp_dir + 'current_params.npy'
    np.save(filename_tmp,params)
    print('Saved params')
    # now we wait for the go
    filename_go_stimulus = tmp_dir + 'go_stimulus.npy'
    
    wait = 1
    while wait:
        if os.path.isfile(filename_go_stimulus):
            # we got the go signals from the online analysis computer!! Hurra!
            wait = 0
            # we remove the go signal
            os.remove(filename_tmp)
            os.remove(filename_go_stimulus)
        else:
            time.sleep(0.5)
            print('Wait for Go!')
            
def generate_filename_and_make_folders(params):
    # %
    date = params['header']['date']
    penetration = params['header']['penetration']
    label = params['header']['label']
    stimulus = params['stimulus']['type']
    
    #filename_tmp =  params['header']['folder_tmp']+'current_params.npy'
    
    folder_save = params['header']['folder_save']+date
    if not params['test']: # if we test we dont make dirs etc. it just takes time
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)
    
    filename_save = folder_save+'\\'+penetration+label+stimulus+'_params.npy'
    
    return filename_save
    

# %% Use the code above to generate images %% #

def present_images(params,setup):
     # %% we create some filenames, the tmp is used for the communication with the stimulus pc. the save for saving ... haha 
    filename_save = OurSetup.generate_filename_and_make_folders(params)
    # %% Create window
    win = setup['win']
    # trigger = setup['trigger']
    background_color = params['monitor']['background']
    background_color
    win.setRGB(background_color)
    win.flip()
    # win,trigger = OurSetup.OpenScreen(background_color,monitor_distance)
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
    # %%
    np.random.seed(1675)
    
    xn = 40
    yn = 40
    
    checker = np.random.randint(2,size=(xn,yn)).astype(float)
    checker *= 2.
    checker -= 1.
    
    checker_inverse = checker*-1.
    # %%
    #n_frames = stimulus_frames.shape[2]
    stim_size = np.array([xn,yn]).astype('float')
    #images = {}
    image_checker = visual.ImageStim(win,checker,size=stim_size*scale,pos=position)
    image_checker_inverse = visual.ImageStim(win,checker_inverse,size=stim_size*scale,pos=position)
    # %%   
    
    # we save the params
    if not params['test']:
        # we save the params in the folder, always needed. Just in test maybe not
        np.save(filename_save,params)
        # we wait for the online analysis
        if params['closed_loop_with_analysis']:
            OurSetup.write_current_params_and_wait_for_go(params)
    
    # we present some text outside the screen to get the system up and running
    # OurSetup.present_pause(120,win,trigger)# in frames    
    
    for trial in range(int(params['stimulus']['trials'])):
        # on even trials we present checker, on odds checker inverse
        if np.mod(trial,2) == 0: # even 
            image = image_checker
        else:
            image = image_checker_inverse
                
        # present images
        for flipN in range(stimulus_duration_in_frames):
            image.draw()
            win.flip()
            #raw_input()
            if flipN == 0:
                # trigger.FrameTime()
#            trigger.EyeCamera()
                a = 1
    # done
    # we flip one more time to make the screen gray
    win.flip()
    # OurSetup.present_pause(360,win,trigger)# in frames
    
    # %%
    #OurSetup.present_pause(240,win)# in frames
    
    # trigger.Stimulus_Stop()
    #win.close()



def present_images_with_first_frame_trigger(stimulus_frames,params,setup):
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
    # %% generate images
    scale = params['stimulus']['scale']
    position = params['stimulus']['position']
    stimulus_duration_in_frames = params['stimulus']['stimulus_duration_in_frames']
    n_frames = stimulus_frames.shape[2]
    stim_size = np.array([stimulus_frames.shape[1],stimulus_frames.shape[0]]).astype('float')
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
                    trigger.FrameTime()
        # we flip one more time to make the screen gray
        win.flip()
        OurSetup.present_pause(360,win)# in frames
    
    # %%
    #OurSetup.present_pause(240,win)# in frames
    
    trigger.Stimulus_Stop()



def present_images_with_delay(stimulus_frames,scale,frame_durations,delay_duration,position,background_color,n_protocol,monitor_distance):
    # open screen
    win,trigger = OurSetup.OpenScreen(background_color,monitor_distance)
    trigger.Stimulus_Start()
    
    # %% generate images
    n_frames = stimulus_frames.shape[2]
    stim_size = np.array([stimulus_frames.shape[1],stimulus_frames.shape[0]]).astype('float')
    images = {}
    for frameN in range(n_frames):
        images[str(frameN)]=visual.ImageStim(win,stimulus_frames[:,:,frameN],size=stim_size*scale,pos=position)
    
    
    # protocol
    for k in range(n_protocol):
        trigger.triggger_protocol()
        time.sleep(0.1)
       
    n_trials = 1
    
    stim=visual.TextStim(win)
    countdown = 20*2
    for i in range(countdown):
        stim.text = str(countdown-i)
        stim.pos = [0,0]
        stim.draw()
        win.flip()
    
    # present images
    for ti in range(n_trials):
        for stim_n in range(len(frame_durations)):
            frame_duration = frame_durations[stim_n]
            for flipN in range(frame_duration):
                images[str(frameN)].draw()
                win.flip()
                if flipN == 0:
                    trigger.FrameTime()
        # after we have presented the stimulus we will wait a for the duration o delay. we have to present something for reliable frames
        for i in range(delay_duration):
            stim.text = 'delay'
            stim.pos = [-2000,0] # should not be visible
            stim.draw()
            win.flip()
    
    # we flip one more time to make the screen gray
    win.flip()
    
    # %%
    trigger.Stimulus_Stop()
    #win.close()



def present_images_with_delay_lr(stimulus_frames,scale,frame_duration,position,background_color,n_protocol,delay_duration,monitor_distance):
    # open screen
    win,trigger = OurSetup.OpenScreen(background_color,monitor_distance)
    trigger.Stimulus_Start()
    
    # %% generate images
    n_frames = stimulus_frames.shape[2]
    stim_size = np.array([stimulus_frames.shape[1],stimulus_frames.shape[0]]).astype('float')
    images = {}
    for frameN in range(n_frames):
        images[str(frameN)]=visual.ImageStim(win,stimulus_frames[:,:,frameN],size=stim_size*scale,pos=position)
    
    
    # protocol
    for k in range(n_protocol):
        trigger.triggger_protocol()
        time.sleep(0.1)
       
    n_trials = 1
    
    stim=visual.TextStim(win)
    countdown = 20*2
    for i in range(countdown):
        stim.text = str(countdown-i)
        stim.pos = [0,0]
        stim.draw()
        win.flip()
    
    # present images
    for ti in range(n_trials):
        for frameN in range(n_frames):
            
            for flipN in range(frame_duration):
                images[str(frameN)].draw()
                win.flip()
                if flipN == 0:
                    trigger.FrameTime()
            # after we have presented the stimulus we will wait a for the duration o delay. we have to present something for reliable frames
            for i in range(delay_duration):
                stim.text = 'delay'
                stim.pos = [-2000,0] # should not be visible
                stim.draw()
                win.flip()
    # we flip one more time to make the screen gray
    win.flip()
    stop_duration = 500
    for i in range(stop_duration):
                stim.text = 'delay'
                stim.pos = [-2000,0] # should not be visible
                stim.draw()
                win.flip()
    
    # %%
    trigger.Stimulus_Stop()
    #win.close()