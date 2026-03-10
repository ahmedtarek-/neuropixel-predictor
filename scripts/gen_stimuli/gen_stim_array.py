"""
Created on Mon Sep 15 17:07:46 2025

Generates a stimuli matrix numpy file and an optional video.

- Can be used to generate:
    - Moving Bars
    - Moving Gratings
    - Sparse Noise Dark  (on light background)
    - Sparse Noise Light (on dark background)

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


# def getMovingStims(
#         orientRads, xLenPx, yLenPx, xLenDeg, yLenDeg, frameRate, trialDurationSec, 
#         waveLenDegs, temporalFreqs, peakLuminModulation=1, bgLumin=0, 
#         centerXY=[0,0], pauseDurationSec=0.5, pauseLumin=0, useBar=False):
#     """
#     PARAMETERS
#     ----------
#     pauseDurationSec : The pasue duration between trials in second.
#     pauseLumin : The pause luminance. Black = -1, gray = 0, white = 1.
#     """
#     xDegPerPx = xLenDeg / xLenPx    # Increments of x
#     yDegPerPx = yLenDeg / yLenPx    # Increments of y
#     xsDeg = np.linspace(xDegPerPx, xLenDeg, xLenPx)
#     ysDeg = np.linspace(yDegPerPx, yLenDeg, yLenPx)
#     frameDurationSec = 1 / frameRate
#     frameTimesSec = np.arange(0, trialDurationSec, frameDurationSec)
#     nPauseFrame = int(round(pauseDurationSec / frameDurationSec))
#     pauseFrames = np.ones((nPauseFrame, yLenPx, xLenPx)) * pauseLumin

#     print("xLenDeg: ", xLenDeg)
#     print("xLenPx: ", xLenPx)
#     print("yLenDeg: ", yLenDeg)
#     print("yLenPx: ", yLenPx)
#     print("xDegPerPx: ", xDegPerPx)
#     print("yDegPerPx: ", yDegPerPx)
#     # print("xsDeg: ", xsDeg)
#     # print("ysDeg: ", ysDeg)
#     # print("frameDurationSec: ", frameDurationSec)
#     print("frameTimesSec: ", frameTimesSec)
#     print("total frames: ", len(frameTimesSec))
#     print("nPauseFrame: ", nPauseFrame)
#     print("pauseFrames: ", pauseFrames)
#     print("waveLenDegs: ", waveLenDegs)
#     print("temporalFreqs: ", temporalFreqs)
    
#     for i, ori in enumerate(orientRads):
#         movingStim = getMovingStim(
#             xsDeg, ysDeg, frameTimesSec, ori, waveLenDegs[i], temporalFreqs[i], 
#             peakLuminModulation, bgLumin, centerXY, useBar)
#         movingStims = movingStim if i == 0 else np.vstack((movingStims, movingStim))
#         movingStims = np.vstack((movingStims, pauseFrames))
#     return movingStims

# def getMovingStim(
#         xsDeg, ysDeg, frameTimesSec, orientRad=0, waveLenDeg=10, temporalFreq=2, 
#         peakLuminModulation=1, bgLumin=0, centerXY=[0,0], useBar=False):
#     """
#     RETURN
#     ------
#     movingStim : array_like, 3-D
#         The generated moving stimuli. Shape = (nFrame, ny, nx).
#     """
#     movingStim = []
#     for t in frameTimesSec:
#         if useBar:
#             # stim = genBar2D(
#             #     xsDeg, ysDeg, t, orientRad, waveLenDeg, temporalFreq, 
#             #     peakLuminModulation, bgLumin, centerXY)
#             stim = genBar2D(
#                 xsDeg, ysDeg, t,
#                 orientRad=orientRad,
#                 barWidthDeg=waveLenDeg,
#                 speedDegPerSec=150
#             )
#         else:
#             stim = genGratings2D(
#                 xsDeg, ysDeg, t, orientRad, waveLenDeg, temporalFreq, 
#                 peakLuminModulation, bgLumin, centerXY)
#         movingStim.append(stim)
#     return np.array(movingStim)

def getMovingStim(
        orientRad, xLenPx, yLenPx, xLenDeg, yLenDeg, frameRate, trialDurationSec, 
        waveLenDeg, temporalFreq, peakLuminModulation=1, bgLumin=0, 
        centerXY=[0,0], pauseDurationSec=0.5, pauseLumin=0, useBar=False):
    """
    PARAMETERS
    ----------
    pauseDurationSec : The pasue duration between trials in second.
    pauseLumin : The pause luminance. Black = -1, gray = 0, white = 1.
    """
    xDegPerPx = xLenDeg / xLenPx    # Increments of x
    yDegPerPx = yLenDeg / yLenPx    # Increments of y
    xsDeg = np.linspace(xDegPerPx, xLenDeg, xLenPx)
    ysDeg = np.linspace(yDegPerPx, yLenDeg, yLenPx)
    frameDurationSec = 1 / frameRate
    frameTimesSec = np.arange(nFrames) / frameRate
    nPauseFrame = int(round(pauseDurationSec / frameDurationSec))
    pauseFrames = np.ones((nPauseFrame, yLenPx, xLenPx)) * pauseLumin

    
    movingStim = []
    for t in frameTimesSec:
        if useBar:
            # stim = genBar2D(
            #     xsDeg, ysDeg, t,
            #     orientRad=orientRad,
            #     barWidthDeg=waveLenDeg,
            #     speedDegPerSec=150
            # )
            stim = genBar2D(
                xsDeg,
                ysDeg,
                frameN=frame,
                orientDeg=90,
                barWidthDeg=10,
                barLengthDeg=240,
                speedDegPerSec=150,
                frameRate=120,
                radiusDeg=120,
                peakLumin=1,
                bgLumin=-1
            )
        else:
            stim = genGratings2D(
                xsDeg, ysDeg, t, orientRad, waveLenDeg, temporalFreq, 
                peakLuminModulation, bgLumin, centerXY)
        movingStim.append(stim)
    return np.array(movingStim)

# def genBar2D(
#         xs, ys, t, orientRad=0, barWidthDeg=10, speedDegPerSec=150,
#         peakLuminModulation=1, bgLumin=-1, centerXY=[0,0],
#         radius=120):

#     # xc0 = np.cos(orientRad) * radius * -1
#     # yc0 = -np.sin(orientRad) * radius * -1

#     # vx = (-np.cos(orientRad)*-1.) * speedDegPerSec
#     # vy = ( np.sin(orientRad)*-1.) * speedDegPerSec
#     xc0 = -np.cos(orientRad) * radius
#     yc0 = -np.sin(orientRad) * radius

#     vx =  np.cos(orientRad) * speedDegPerSec
#     vy =  np.sin(orientRad) * speedDegPerSec

#     xc = xc0 + vx * t
#     yc = yc0 + vy * t

#     xs, ys = np.meshgrid(xs, ys)

#     # rotate coordinate system to bar frame
#     xRot = (xs-xc)*np.cos(orientRad) + (ys-yc)*np.sin(orientRad)

#     bar = np.ones_like(xs)*bgLumin

#     mask = np.abs(xRot) <= barWidthDeg/2

#     bar[mask] = peakLuminModulation

#     return bar
def genBar2D(
    xsDeg, ysDeg,
    frameN,
    orientDeg,
    barWidthDeg,
    barLengthDeg,
    speedDegPerSec,
    frameRate,
    radiusDeg,
    peakLumin=1,
    bgLumin=-1
):
    """
    Exact geometric equivalent of PsychoPy ImageStim moving bar.

    Parameters
    ----------
    xsDeg, ysDeg : 1D arrays
        Screen coordinates in degrees
    frameN : int
        Frame index (0-based)
    orientDeg : float
        ORIGINAL PsychoPy orientation in degrees (before remap)
    barWidthDeg : float
    barLengthDeg : float
    speedDegPerSec : float
    frameRate : float
    radiusDeg : float
    peakLumin : float
    bgLumin : float
    """

    # --- PsychoPy orientation remap ---
    theta = np.deg2rad(360.0 - orientDeg)

    # --- PsychoPy starting position ---
    x0 = np.cos(theta) * radiusDeg
    y0 = -np.sin(theta) * radiusDeg

    x0 *= -1
    y0 *= -1

    # --- PsychoPy per-frame displacement ---
    speedPerFrame = speedDegPerSec / frameRate

    dx = (-np.cos(theta)*-1.) * speedPerFrame
    dy = ( np.sin(theta)*-1.) * speedPerFrame

    # --- Position at this frame ---
    xc = x0 + dx * frameN
    yc = y0 + dy * frameN

    # --- Meshgrid ---
    xs, ys = np.meshgrid(xsDeg, ysDeg)

    # --- Rotate coordinates into bar frame ---
    x_rel = xs - xc
    y_rel = ys - yc

    x_rot =  x_rel * np.cos(theta) + y_rel * np.sin(theta)
    y_rot = -x_rel * np.sin(theta) + y_rel * np.cos(theta)

    # --- Rectangle mask ---
    mask = (
        (np.abs(x_rot) <= barWidthDeg/2) &
        (np.abs(y_rot) <= barLengthDeg/2)
    )

    # --- Create image ---
    img = np.full(xs.shape, bgLumin, dtype=np.float32)
    img[mask] = peakLumin

    return img

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

# orientRads = [0.75*np.pi, 0, np.pi]
orientations = [90]
# orientRads = [o * np.pi / 180 for o in orientations]
orientRads = [(360-o) * np.pi / 180 for o in orientations]
xLenPx = 36*10
yLenPx = 22*10
xLenDeg = 120
yLenDeg = 90
frameRate = 120
nFrames = 192
trialDurationSec = 1.5
spatialFreq = np.array([0.08, 0.01, 0.04])   # cyclePerDeg
waveLenDegs = 1 / spatialFreq
temporalFreqs = [10] * len(orientRads)
barWidthDegs = [10] * len(orientRads)
peakLuminModulation = 1
pauseDurationSec = 0
pauseLumin = 0

useBar = True
bgLumin = -1 if useBar else 0
widthDegs = barWidthDegs if useBar else waveLenDegs

movingStims = getMovingStim(
    orientRads[0], xLenPx, yLenPx, xLenDeg, yLenDeg, frameRate, nFrames, 
    widthDegs[0], temporalFreqs[0], peakLuminModulation, bgLumin, centerXY=[0,0], 
    pauseDurationSec=pauseDurationSec, pauseLumin=pauseLumin, useBar=useBar)

folder_path = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli'
stim_name = 'bar' if useBar else 'grating'
moving_file_name = f"moving_{stim_name}.npy"
moving_file_path = os.path.join(folder_path, moving_file_name)

print(movingStims.shape)
with open(moving_file_path, 'wb') as f:
    np.save(f, movingStims)

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
# print(stimArr.shape)
# folder_path = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli'
# file_name = 'sparse_noise_dark_36_22.npy'
# file_path = os.path.join(folder_path, file_name)
# stimArr_reshaped_to_xy = np.moveaxis(stimArr, -1, 1)
# with open(file_path, 'wb') as f:
#     np.save(f, stimArr_reshaped_to_xy)

####%% Generate and save video %%####

saveMovingStim = True

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
