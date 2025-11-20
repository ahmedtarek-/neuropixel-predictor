# Scripts

## 1. Calculating firing rate `calculate_firing_rate.py`

Calculates the firing rate :D

## 2. Generate stimulus Numpy array `gen_stim_array.py`

- Can be used to generate:
    - Moving Bars
    - Moving Gratings
    - Sparse Noise Light (on dark background)
    - Sparse Noise Dark  (on light background)

## 3. Generate stimulus Numpy array `gen_checkerboard.py`
Generates the checkerboard stimulus matrix. 

## 4. Generate training and test data
Given stimuli files and firing rate files generated from last two steps, this script
bundles all data together and shuffle it together then split into training and test data.
