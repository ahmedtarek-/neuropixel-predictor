# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 07:19:12 2018

@author: AGKremkow
"""
# %%
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
    
    

# %%
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
        
    def Stimulus_Stop(self):
#        self.triggger(4) # pin 3 is 4 in dec 001000 (WARNING BINARY CODE HERE)
        self.triggger(16) # pin 3 is 4 in dec 001000 (WARNING BINARY CODE HERE)
        # now we write the params to the folder TO BE UNCOMMENT
#        folder_save = params['header']['folder_finished_protocols']
#        protocol_counter = params['header']['protocol_counter']
#        date = params['header']['date']
#        penetration = params['header']['penetration']
#        #label = params['header']['label']
#        stimulus = params['stimulus']['type']
#        filename_save = folder_save+'\\'+date+'_'+str(protocol_counter).zfill(3)+'_'+penetration+'_'+stimulus+'_params.npy'
#        np.save(filename_save,params)
#    
#    def triggger_protocol(self):
#        duration_local = 0.002
#        self.port.Out32(self.portadress,8)
#        core.wait(duration_local)
#        self.port.Out32(self.portadress,0)
        #TO BE UNCOMMENT
        
        
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
        DellMonitor = monitors.Monitor('') # to be updated
        DellMonitor.setDistance(monitor_distance)
        win= visual.Window(monitor = DellMonitor,screen=0,waitBlanking=True)#,size=[1512, 982],fullscr=True,allowGUI=False,units='deg',color=background_color)
        print('dell screen options selected')
        
          
    # trigger = Trigger()
    time.sleep(1.)
    
    return win


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
