"""
Given a single unit file name, calculates firing rate.

Removes the units that are 'Flags' or 'Dendrites' (based on manual classification)
{'MPW-Dendrite', 'Flag', 'MPW-Axon', 'SU-Regular', 'SU-Small', 'SU-Fast'}


Things that needs to be changed:
    1. 'experiment_name': Single unit file.
    2. 'stimulus_name': Which stimulus to calculate FR for.
    3. 'stimulus': Path to file that contains the actual stimulus used.
    4. 'save_stim_file_name': Name of file to save stimulus.
    5. 'save_fr_file_name': Name of file to save firing rates


Stimuli files processed so far:
    [sparse_noise_light_36_22.npy, sparse_noise_dark_36_22.npy, checkerboard_200.npy]

Hints:
    - `data["stim_params_files"]["mb"]["stimulus"]["sequence"]["orientations"]`
    should match number of TTls in that protocol.

    - `data["stim_params_files"]["csd"]["stimulus"]["trials"]` gives number
    of trials in the checkerboard.

@author: Ahmed Abdalfatah (@ahmedtarek-)
"""

import os
import numpy as np
import pickle

NP_FRQ = 30_000
EXCLUDE_UNITS = ['Flag', 'MPW-Dendrite']

# Choose stimulus (ex. Sd36x22_l_3, Sl36x22_d_3, csd)
stimulus_name = 'Sl36x22_d_3'

# Loading the file
single_unit_folder = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/data-single-unit'
experiment_name = '2022-12-20_15-08-10_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl'
file_path = os.path.join(single_unit_folder, experiment_name)

# Load pickle
import pandas as pd
data = pd.read_pickle(file_path)

# Reshape because of weird shape
data = data.reshape((1))[0]

# To get which stimulius are there
data['events'].keys()
stimulus_ttls = data['events'][stimulus_name]

# Stimulus Length
stimulus_length = len(stimulus_ttls) - 1

# To get spiketimes aligned
st_aligned = data['spiketimes_aligned']

# Clean all units that are in EXCLUDE_UNITS
clean_st_aligned = [
    st for i, st in enumerate(st_aligned)
    if data['classif_from_GUI']['Classification'][i] not in EXCLUDE_UNITS
]

# Calculating spike count per neuron within every frame for the stimulus
firing_rates = []
for first_ttl, second_ttl in zip(stimulus_ttls[:-1], stimulus_ttls[1:]):
    ttl_fr = []
    for neuron in clean_st_aligned:
        spike_count = np.where((neuron >= first_ttl) & (neuron < second_ttl), True, False).sum()
        # Duration: diff_in_ttl / 30000
        # Rate: (sum/duration)
        firing_rate = (spike_count * NP_FRQ) / (second_ttl - first_ttl)
        ttl_fr.append(firing_rate)

    firing_rates.append(ttl_fr)

# Shape (stimulus_length, num_of_units)
fr_per_stimulus = np.array(firing_rates)

# Putting the frames and the firing rates together (Need to match type of stimulus chosen)
stimulus = np.load('/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli/sparse_noise_light_36_22.npy')

# Trim the extra frames.
stimulus = stimulus[:stimulus_length]

# Save the data together (1 -> checkerboard, 2 -> sn dark, 3 -> sn light)
TRAINING_DATA_DIR = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli-Responses'
exp_date = '2022-12-20_15-08'
save_stim_file_name = "{}_3_stimulus_sn_light.npy".format(exp_date)
save_fr_file_name = "{}_3_fr_sn_light.npy".format(exp_date)
with open(os.path.join(TRAINING_DATA_DIR, save_stim_file_name), 'wb') as f:
    np.save(f, stimulus)

with open(os.path.join(TRAINING_DATA_DIR, save_fr_file_name), 'wb') as f:
    np.save(f, fr_per_stimulus)

print("Created ", save_stim_file_name)
print("Created ", save_fr_file_name)
