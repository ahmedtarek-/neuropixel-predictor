# Scripts

The scripts should be run in the following order to end up with data ready for training.

## 1. Generate stimuli arrays

### 1.1 Generate stimulu using `stimuli/run_stimulus.py`
1. Adjust size of screen to get the desired size (ex. 36x22)
2. Change the `stimuli` variable  (ex. 'mb')
3. Adjust the STIMULI_FOLDER in `MovingBarStimulus` (or another file)
4. Run `python stimuli/run_stimulus.py`

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
