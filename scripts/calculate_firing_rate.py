"""
Given a single unit file name, calculates firing rate.

Removes the units that are 'Flags' or 'Dendrites' (based on manual classification)
{'MPW-Dendrite', 'Flag', 'MPW-Axon', 'SU-Regular', 'SU-Small', 'SU-Fast'}


Things that needs to be changed:
    1. 'experiment_name': Single unit file.
    2. 'stimulus_name': Which stimulus to calculate FR for.
    3. 'stimulus': Path to file that contains the actual stimulus used.
    4. 'exp_date': experiment data to be used in file saving
    5. 'save_stim_file_name': Name of file to save stimulus.
    6. 'save_fr_file_name': Name of file to save firing rates


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
import pandas as pd
import pickle

NP_DELAY = 50
NP_FRQ = 30_000
EXCLUDE_UNITS = ['Flag', 'MPW-Dendrite']

MLI_THRESHOLD = 0.7 # Include only top 70% based on modulation index
MLI_FILE = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/MLI_MB/storage_MLI_sparse_noise.npy'

def mli_good_indices(experiment_name):
    """
    Given an experiment, returns the top MLI_THRESHOLD based on
    modulation index of sparse noise dark.
    """
    mli_file = np.load(MLI_FILE, allow_pickle=True, encoding='latin1').item()

    mli = mli_file[experiment_name]['Sd36x22_l_3']['local_MLI'] # Sd36x22_l_3_3
    abs_mli = np.abs(mli)
    indices = abs_mli.argsort()                                 # Returns indices that would sort mli ascendingly.

    truncate_index = int((1 - MLI_THRESHOLD) * len(mli))        # Determine where to cut
    good_indices = indices[truncate_index:-1].astype(np.int_)   # Cut lowest MLI_THRESHOLD and cast as int

    return good_indices

# Choose stimulus (ex. Sd36x22_l_3, Sl36x22_d_3, csd, mb)
stimulus_name = 'Sd36x22_l_3'

# Loading the file
single_unit_folder = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/data-single-unit'
# experiment_name = '2023-03-15_11-05-00_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl'
experiment_name = '2023-03-15_15-23-14_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl'
# experiment_name = '2022-12-20_15-08-10_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl'
file_path = os.path.join(single_unit_folder, experiment_name)

# Modulation index good indices
mli_good_indices = mli_good_indices(experiment_name.split('_Complete')[0])

# Load pickle
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

# To get the cluster ids (unique and important to track)
cluster_ids = data['classif_from_GUI']['clusterIds']

# Clean all units that are in EXCLUDE_UNITS
clean_st_aligned = [
    st for i, st in enumerate(st_aligned)
    if data['classif_from_GUI']['Classification'][i] not in EXCLUDE_UNITS
    and i in mli_good_indices
]

clean_cluster_ids = [
    cluster_id for i, cluster_id in enumerate(cluster_ids)
    if data['classif_from_GUI']['Classification'][i] not in EXCLUDE_UNITS
    and i in mli_good_indices
]

# Dealing with delay by subtracting 50ms from all units in clean_st_aligned.
clean_st_aligned = [(unit - NP_DELAY) for unit in clean_st_aligned]

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
stimulus = np.load('/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli/Psychopy-36x22/sparse_noise_dark_36_22.npy')

# stimulus = np.load('/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli/Psychopy-64x36/sparse_noise_dark_64_36.npy')
# stimulus = np.load('/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli/sparse_noise_light_36_22.npy')
# stimulus = np.load('/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli/checkerboard_200.npy')

# Trim the extra frames.
stimulus = stimulus[:stimulus_length]

# Save the data together (1 -> checkerboard, 2 -> sn dark, 3 -> sn light, 4 -> mb)
TRAINING_DATA_DIR = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli-Responses-36-22'
# exp_date = '2023-03-15_11-05'
exp_date = '2023-03-15_15-23'
# exp_date = '2022-12-20_15-08'

save_stim_file_name = "{}_2_stimulus_sn_dark.npy".format(exp_date)
save_fr_file_name = "{}_2_fr_sn_dark.npy".format(exp_date)
save_cluster_ids_file_name = "{}_cluster_ids.npy".format(exp_date)

with open(os.path.join(TRAINING_DATA_DIR, save_stim_file_name), 'wb') as f:
    np.save(f, stimulus)

with open(os.path.join(TRAINING_DATA_DIR, save_fr_file_name), 'wb') as f:
    np.save(f, fr_per_stimulus)

with open(os.path.join(TRAINING_DATA_DIR, save_cluster_ids_file_name), 'wb') as f:
    np.save(f, np.array(clean_cluster_ids))

print("Created ", save_stim_file_name)
print("Created ", save_fr_file_name)
print("Created ", save_cluster_ids_file_name)
