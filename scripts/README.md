# Scripts

The scripts should be run in the following order to end up with data ready for training.

## 1. Generate stimuli arrays

### 1.1 Generate stimuli using `stimuli/run_stimulus.py`
1. Adjust size of screen to get the desired size (ex. 36x22)
    - In `stimuli/our_setup_new`
    - In `stimuli/ImageStimulus`, `stimuli/MovingBarStimulus` and other files
2. Make sure to use only WIDTH and HEIGHT if not retina. Otherwise size should be halfed
3. Change the `stimuli` variable in `stimuli/run_stimulus`  (ex. 'mb')
4. Adjust the save folder STIMULI_FOLDER in `MovingBarStimulus` (or `ImageStimulus`)
5. Run `python stimuli/run_stimulus.py`

### 1.2 stimulus Numpy array `gen_stim_array.py`

- Can be used to generate:
    - Moving Bars
    - Moving Gratings
    - Sparse Noise Light (on dark background)
    - Sparse Noise Dark  (on light background)

### 1.3 Generate stimulus Numpy array `gen_checkerboard.py`
Generates the checkerboard stimulus matrix. 


## 2. Calculating firing rate `calculate_firing_rate.py`

Calculates the firing rate :D


## 3. Generate training and test data
Given stimuli files and firing rate files generated from last two steps, this script
bundles all data together and shuffle it together then split into training and test data.
