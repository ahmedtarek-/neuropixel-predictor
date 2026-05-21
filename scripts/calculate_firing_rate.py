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
import re
import numpy as np
import pandas as pd
import pickle

######## 1. Defining Constants ########
NP_FRQ = 30_000
NP_DELAY = (50 * NP_FRQ) / 1000

EXCLUDE_UNITS = ['Flag', 'MPW-Dendrite', 'SU-Small', 'SU-Positive']

MLI_THRESHOLD = 0.5 # Include only units with more thab 50% modulation index.
MLI_FILE = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/MLI_MB/storage_MLI_sparse_noise.npy'


######## 2. Choosing stimuli and experiments ########
# Choose stimulus (ex. Sd36x22_l_3, Sl36x22_d_3, csd, mb)
stimulus_name = 'Sd36x22_l_3'

# Loading the file
single_unit_folder = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/data-single-unit'

# Choosing experiment name(s)
experiment_name = '2023-03-15_11-05-00_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl'
# experiment_name = '2023-03-15_15-23-14_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl'
# experiment_name = '2022-12-20_15-08-10_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl'


######## 3. Defining helper method for modulation index based filtering ########
def mli_good_indices(experiment_name):
    """
    Given an experiment, returns the top MLI_THRESHOLD based on
    modulation index of sparse noise dark.
    """
    mli_file = np.load(MLI_FILE, allow_pickle=True, encoding='latin1').item()

    mli = mli_file[experiment_name]['Sd36x22_l_3']['local_MLI'] # Sd36x22_l_3_3
    abs_mli = np.abs(mli)
    good_indices = np.where(abs_mli > MLI_THRESHOLD) # Cut lowest MLI_THRESHOLD and cast as int

    return set(good_indices[0])

######## 4. Extracting experiment date, reading data and preparing variables ########
# Defining pattern to extract right date (notice that the capture group finishes before the last two digits)
pattern = '^([0-9\\-]+\\_[0-9]{2}\\-[0-9]{2})\\-[0-9]{2}\\_Complete.+$'
exp_date = re.search(pattern, experiment_name).groups()[0]
print("exp_date: ", exp_date)

# Modulation index good indices
mli_good_indices = mli_good_indices(experiment_name.split('_Complete')[0])

# Load pickle
file_path = os.path.join(single_unit_folder, experiment_name)
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

print("Number of units before cleaning: ", len(st_aligned))
print("Number of units after MLI cleaning: ", len(mli_good_indices))

######## 5. Cleaning Data ########

# Clean all units that are in EXCLUDE_UNITS or have Modulation index less than threshold.
clean_st_aligned = [
    st for i, st in enumerate(st_aligned)
    if i in mli_good_indices
    and data['classif_from_GUI']['Classification'][i] not in EXCLUDE_UNITS
]

clean_cluster_ids = [
    cluster_id for i, cluster_id in enumerate(cluster_ids)
    if i in mli_good_indices
    and data['classif_from_GUI']['Classification'][i] not in EXCLUDE_UNITS
]
print("Number of units after MLI + classif_from_GUI cleaning: ", len(clean_cluster_ids))

######## 6. Calculate Firing Rate ########

# Dealing with delay by subtracting 50ms from all units in clean_st_aligned. (Retina Delay)
clean_st_aligned = [(unit - NP_DELAY) for unit in clean_st_aligned]

# Calculating spike count per neuron within every frame for the stimulus
firing_rates = []
for first_ttl, second_ttl in zip(stimulus_ttls[:-1], stimulus_ttls[1:]):
    ttl_fr = []
    for neuron in clean_st_aligned:
        spike_count = np.where((neuron >= first_ttl) & (neuron < second_ttl), True, False).sum()
        # Duration: diff_in_ttl / 30_000
        # Rate: (sum/duration)
        firing_rate = (spike_count * NP_FRQ) / (second_ttl - first_ttl)
        ttl_fr.append(firing_rate)

    firing_rates.append(ttl_fr)

# Shape (stimulus_length, num_of_units)
fr_per_stimulus = np.array(firing_rates)

######## 7. Putting Firing Rates with Stimuli Frames ########

# Putting the frames and the firing rates together (Need to match type of stimulus chosen) 
stimulus = np.load('/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli/Psychopy-36x22/sparse_noise_dark_36_22.npy')
# stimulus = np.load('/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli/Psychopy-64x36/sparse_noise_dark_64_36.npy')
# stimulus = np.load('/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli/sparse_noise_light_36_22.npy')
# stimulus = np.load('/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli/checkerboard_200.npy')

# Trim the extra frames.
stimulus = stimulus[:stimulus_length]

# Save the data together (1 -> checkerboard, 2 -> sn dark, 3 -> sn light, 4 -> mb)
TRAINING_DATA_DIR = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli-Responses-36-22'

save_stim_file_name = "{}_2_stimulus_sn_dark.npy".format(exp_date)
save_fr_file_name = "{}_2_fr_sn_dark.npy".format(exp_date)
save_cluster_ids_file_name = "{}_cluster_ids.npy".format(exp_date)

with open(os.path.join(TRAINING_DATA_DIR, save_stim_file_name), 'wb') as f:
    np.save(f, stimulus)

with open(os.path.join(TRAINING_DATA_DIR, save_fr_file_name), 'wb') as f:
    np.save(f, fr_per_stimulus)

with open(os.path.join(TRAINING_DATA_DIR, save_cluster_ids_file_name), 'wb') as f:
    np.save(f, np.array(clean_cluster_ids))

print("------")
print("Created ", save_stim_file_name)
print("Created ", save_fr_file_name)
print("Created ", save_cluster_ids_file_name)
