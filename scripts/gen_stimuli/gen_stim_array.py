"""
Created on Mon Sep 15 17:07:46 2025

Generates a stimuli matrix numpy file and an optional video.

- Can be used to generate:
    - Moving Bars
    - Moving Gratings
    - Sparse Noise Light (on dark background)
    - Sparse Noise Dark  (on light background)

- Don't create checkerboard using this because it needs to be cropped first.
    Use gen_checkerboard.py instead


@author: kailun
"""
import os
import numpy as np
from scipy import signal, ndimage
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter


def getMovingStims(
        orientRads, xLenPx, yLenPx, xLenDeg, yLenDeg, frameRate, trialDurationSec, 
        waveLenDegs, temporalFreqs, peakLuminModulation=1, bgLumin=0, 
        centerXY=[0,0], pauseDurationSec=0.5, pauseLumin=0, useBar=False):
    """
    PARAMETERS
    ----------
    pauseDurationSec : The pasue duration between trials in second.
    pauseLumin : The pause luminance. Black = -1, gray = 0, white = 1.
    """
    xDegPerPx = xLenDeg / xLenPx
    yDegPerPx = yLenDeg / yLenPx
    xsDeg = np.linspace(xDegPerPx, xLenDeg, xLenPx)
    ysDeg = np.linspace(yDegPerPx, yLenDeg, yLenPx)
    frameDurationSec = 1 / frameRate
    frameTimesSec = np.arange(0, trialDurationSec, frameDurationSec)
    nPauseFrame = int(round(pauseDurationSec / frameDurationSec))
    pauseFrames = np.ones((nPauseFrame, yLenPx, xLenPx)) * pauseLumin
    
    for i, ori in enumerate(orientRads):
        movingStim = getMovingStim(
            xsDeg, ysDeg, frameTimesSec, ori, waveLenDegs[i], temporalFreqs[i], 
            peakLuminModulation, bgLumin, centerXY, useBar)
        movingStims = movingStim if i == 0 else np.vstack((movingStims, movingStim))
        movingStims = np.vstack((movingStims, pauseFrames))
    return movingStims

def getMovingStim(
        xsDeg, ysDeg, frameTimesSec, orientRad=0, waveLenDeg=10, temporalFreq=2, 
        peakLuminModulation=1, bgLumin=0, centerXY=[0,0], useBar=False):
    """
    RETURN
    ------
    movingStim : array_like, 3-D
        The generated moving stimuli. Shape = (nFrame, ny, nx).
    """
    movingStim = []
    for t in frameTimesSec:
        if useBar:
            stim = genBar2D(
                xsDeg, ysDeg, t, orientRad, waveLenDeg, temporalFreq, 
                peakLuminModulation, bgLumin, centerXY)
        else:
            stim = genGratings2D(
                xsDeg, ysDeg, t, orientRad, waveLenDeg, temporalFreq, 
                peakLuminModulation, bgLumin, centerXY)
        movingStim.append(stim)
    return np.array(movingStim)

def genBar2D(
        xs, ys, t, orientRad=0, barWidthDeg=10, frequency=2, 
        peakLuminModulation=1, bgLumin=-1, centerXY=[0,0]):
    """To generate the 2-D bar at time t.
    PARAMETERS
    ----------
    xs, ys : array_like, 1-D
        The x- and y-coordinates (in deg) of the axes for generating the bar.
    t : int or float
        The time point (sec) from 0 for moving/shifting the bar.
    orientRad : int or float
        The orientation of the bar in radian.
    barWidthDeg, frequency : int or float
        The bar width in deg and the temporal frequency in Hz for the bar.
    peakLuminModulation, bgLumin : int or float
        The peak amplitude of the modulating luminance and the background luminance (a.u.).
    centerXY : list, tuple, or array_like, 1-D
        The x- and y-coordinates for the center of reference.
    
    RETURN
    ------
    bar : array_like, 2-D
        The generated bar at time t.
    """
    xc, yc = centerXY
    xs, ys = np.meshgrid(xs, ys)
    xDiff = xs - xc
    yDiff = ys - yc
    distances = np.sqrt(xDiff**2 + yDiff**2)
    phase = np.cos(np.arctan2(xDiff, yDiff) + orientRad)
    cycles = frequency * t
    startPhase = distances * phase / barWidthDeg
    normPhase = distances * phase / barWidthDeg - cycles
    barCenter = startPhase.min()
    barMask = (normPhase >= barCenter - 0.5) & (normPhase < barCenter + 0.5)
    bar = bgLumin * np.ones_like(normPhase)
    bar[barMask] = peakLuminModulation
    return bar

def genGratings2D(
        xs, ys, t, orientRad=0, wavelength=10, frequency=2, 
        peakLuminModulation=40, bgLumin=50, centerXY=[0,0]):
    """To generate the 2-D sinusoidal gratings at time t.
    PARAMETERS
    ----------
    xs, ys : array_like, 1-D
        The x- and y-coordinates (in deg) of the axes for generating the gratings.
    t : int or float
        The time point (sec) from 0 for moving/shifting the gratings.
    orientRad : int or float
        The orientation of the gratings in radian.
    wavelength, frequency : int or float
        The wavelength in deg and the temporal frequency in Hz for the gratings.
    peakLuminModulation, bgLumin : int or float
        The peak amplitude of the modulating luminance and the background luminance (a.u.).
    centerXY : list, tuple, or array_like, 1-D
        The x- and y-coordinates for the center of reference.
    
    RETURN
    ------
    gratings : array_like, 2-D
        The generated gratings at time t.
    """
    xc, yc = centerXY
    xs, ys = np.meshgrid(xs, ys)
    xDiff = xs - xc
    yDiff = ys - yc
    distances = np.sqrt(xDiff**2 + yDiff**2)
    phase = np.cos(np.arctan2(xDiff, yDiff) + orientRad)
    cycles = frequency * t
    normPhase = distances * phase / wavelength - cycles
    gratings = peakLuminModulation * np.cos(2*np.pi*normPhase)
    gratings += bgLumin
    return gratings

def expandStimArr(stimArr, xLenPx, yLenPx):
    """
    PARAMETERS
    ----------
    stimArr : 3-D array-like
        The stimulus array to be resized, shape = (nFrame, ny, nx).
    """
    _, ny, nx, = stimArr.shape
    xScale = xLenPx / nx
    yScale = yLenPx / ny
    stimArr = ndimage.zoom(stimArr, (1, yScale, xScale), order=0)
    return stimArr

def saveVideo(
        savePath, frames, figWidth=10, fps=15, bitrate=1800, interval=500, 
        axisOff=True, marginOff=True, originLow=True):
    """
    frames : 3D array-like
        The frames to be saved. Shape = (nFrame, yLen, xLen)
    """
    _, ny, nx = frames.shape
    figHeight = figWidth / nx * ny
    origin = 'lower' if originLow else None
    
    writer = FFMpegWriter(fps=fps, bitrate=bitrate)
    fig = plt.figure(figsize=(figWidth, figHeight))
    ims = []
    if marginOff:
        fig.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = plt.gca()
    if axisOff:
        ax.axis('off')
    for f, frame in enumerate(frames):
        im = [ax.imshow(frame, cmap='gray', origin=origin, vmin=-1, vmax=1)]
        ims.append(im)
    plt.close(fig)
    ani = animation.ArtistAnimation(
        fig, ims, interval=interval, repeat_delay=1000)
    ani.save(savePath, writer=writer)
    plt.close()


##################################################
#### Part 1: Generate moving gratings or bars ####
##################################################

orientRads = [0.75*np.pi, 0, np.pi]
xLenPx = 36
yLenPx = 22
xLenDeg = 120
yLenDeg = 90
frameRate = 12
trialDurationSec = 5
spatialFreq = np.array([0.08, 0.01, 0.04])   # cyclePerDeg
waveLenDegs = 1 / spatialFreq
temporalFreqs = [4] * len(orientRads)
barWidthDegs = [10] * len(orientRads)
peakLuminModulation = 1
pauseDurationSec = 1
pauseLumin = 0
useBar = True
bgLumin = -1 if useBar else 0
widthDegs = barWidthDegs if useBar else waveLenDegs

# movingStims = getMovingStims(
#     orientRads, xLenPx, yLenPx, xLenDeg, yLenDeg, frameRate, trialDurationSec, 
#     widthDegs, temporalFreqs, peakLuminModulation, bgLumin, centerXY=[0,0], 
#     pauseDurationSec=pauseDurationSec, pauseLumin=pauseLumin, useBar=useBar)

# print(movingStims.shape)
# with open('mb_trial.npy', 'wb') as f:
#     np.save(f, movingStims)

########################################################
#### Part 2: Sparse Noise and Checkerboard (resize) ####
########################################################


# stimPath = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli/Checkerboard_12_7_target_size_3_n_frames_500_20200430.npy'
stimPath = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli/locally_dark_sparse_noise_36_22_target_size_3_targets_per_frame_2_trials_10_background_1.0_20181120.npy'
# stimArr = np.load(stimPath, allow_pickle=True, encoding='latin1').item()   # ny,   , nFrame
stimArr = np.load(stimPath, allow_pickle=True, encoding='latin1').item()['frames']   # if the file is dictionary
stimArr = np.moveaxis(stimArr, -1, 0)   # nFrame, ny, nx
stimArr = expandStimArr(stimArr, xLenPx, yLenPx)

# Save file (after moving axis to have (nFrame, nx, ny))
print(stimArr.shape)
folder_path = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli'
file_name = 'sparse_noise_dark_36_22.npy'
file_path = os.path.join(folder_path, file_name)
stimArr_reshaped_to_xy = np.moveaxis(stimArr, -1, 1)
with open(file_path, 'wb') as f:
    np.save(f, stimArr_reshaped_to_xy)

####%% Generate and save video %%####

saveMovingStim = False

if saveMovingStim:
    stimKey = 'bars' if useBar else 'gratings'
    frames = movingStims.copy()
else:
    stimKey = 'sparse_noise_dark'
    frames = stimArr.copy()

savePath = f'/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli/Videos/{stimKey}.mp4'
saveVideo(
    savePath, frames, figWidth=10, fps=frameRate, bitrate=1800, interval=500, 
    axisOff=True, marginOff=True, originLow=True)
