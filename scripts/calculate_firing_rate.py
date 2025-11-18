"""

Given a single unit file name.

TODO: Make saving folder and save filename as variables

@author: Ahmed Abdalfatah (@ahmedtarek-)
"""

import os
import numpy as np
import pickle

NP_FRQ = 30_000
stimulus_name = 'Sd36x22_l_3'

# Loading the file
single_unit_folder = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/data-single-unit'
file_name = '2023-03-15_11-05-00_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl'
file_path = os.path.join(single_unit_folder, file_name)


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

# Calculating spike count per neuron within every frame for the csd
firing_rates = []
for first_ttl, second_ttl in zip(stimulus_ttls[:-1], stimulus_ttls[1:]):
    ttl_fr = []
    for neuron in data['spiketimes_aligned']:
        spike_count = np.where((neuron >= first_ttl) & (neuron < second_ttl), True, False).sum()
        # Duration: diff_in_ttl / 30000
        # Rate: (sum/duration)
        firing_rate = (spike_count * NP_FRQ) / (second_ttl - first_ttl)
        ttl_fr.append(firing_rate)

    firing_rates.append(ttl_fr)

# Shape (stimulus_length, 453)
fr_per_stimulus = np.array(firing_rates)

# Putting the frames and the firing rates together
stimulus = np.load('/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli/sparse_noise_dark_36_22.npy')

# Trim the extra frames.
stimulus = stimulus[:stimulus_length]

# Save the data together
TRAINING_DATA_DIR = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli-Responses'
with open(os.path.join(TRAINING_DATA_DIR, '2_stimulus_sn_d_on_2023-03-15_11-05.npy'), 'wb') as f:
    np.save(f, stimulus)

with open(os.path.join(TRAINING_DATA_DIR, '3_fr_sn_l_on_2023-03-15_11-05.npy'), 'wb') as f:
    np.save(f, fr_per_stimulus)
