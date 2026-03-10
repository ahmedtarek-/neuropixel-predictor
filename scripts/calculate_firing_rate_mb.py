"""
For moving bar and moving grating, calculates firing rate.

Removes the units that are 'Flags' or 'Dendrites' (based on manual classification)
{'MPW-Dendrite', 'Flag', 'MPW-Axon', 'SU-Regular', 'SU-Small', 'SU-Fast'}

Things that needs to be changed:
    1. 'experiment_name': Single unit file.
    2. 'stimulus_name': Which stimulus to calculate FR for.
    3. 'stimulus': Path to file that contains the actual stimulus used.
    4. 'exp_date': experiment data to be used in file saving
    5. 'save_stim_file_name': Name of file to save stimulus.
    6. 'save_fr_file_name': Name of file to save firing rates

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


########### 1. Define variables and data location ###########
DATA_FOLDER = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data'
PSYCHOPY_STIMULI = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli/Psychopy'
TRAINING_DATA_SAVE_DIR = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli-Responses-Delay'

VALIDATION_PLOT = True

NP_DELAY = 50
NP_FRQ = 30_000

NUM_FRAMES = 192
FRAME_SPAN = (1/120) * NP_FRQ

EXCLUDE_UNITS = ['Flag', 'MPW-Dendrite']

# Dates: 2022-12-20_15-08-10, 2023-03-15_11-05-00, 2023-03-15_15-23-14
experiment_date = '2023-03-15_15-23-14'
experiment_tag = experiment_date[:-3]

# Choose stimulus (ex. mb, mg)
stimulus_name = 'mb'

# Define stimuli_params_key
stimuli_params_keys = {
  '2022-12-20_15-08-10': '20221220/c11emb_params.npy',
  '2023-03-15_11-05-00': '20230315bis/c11cmb_params.npy',
  '2023-03-15_15-23-14': '20230315/c11cmb_params.npy',
}

# func_resp_folder = os.path.join(DATA_FOLDER, 'functional-responses')
# func_resp_file = f"functional_responses_{experiment_date}_l.pkl"
# func_resp_file = os.path.join(func_resp_folder, func_resp_file)

# Loading the file
single_unit_folder = os.path.join(DATA_FOLDER, 'data-single-unit')
stim_params_folder = os.path.join(DATA_FOLDER, 'Stimuli-Params')


single_unit_file = f"{experiment_date}_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl"
stim_params_file = stimuli_params_keys[experiment_date]

single_unit_file = os.path.join(single_unit_folder, single_unit_file)
stim_params_file = os.path.join(stim_params_folder, stim_params_file)

########### 2. Prepare data to calculate FR ###########
# Load pickle
data = pd.read_pickle(single_unit_file)

stim_params = np.load(stim_params_file, allow_pickle=True, encoding='latin1').item()

# Reshape because of weird shape
data = data.reshape((1))[0]

# To get which stimulius are there
# data['events'].keys()
stimulus_ttls = data['events'][stimulus_name]

# Stimulus Length
stimulus_length = len(stimulus_ttls)

# To get spiketimes aligned
st_aligned = data['spiketimes_aligned']

# To get the cluster ids (unique and important to track)
cluster_ids = data['classif_from_GUI']['clusterIds']

# Clean all units that are in EXCLUDE_UNITS
clean_st_aligned = [
    st for i, st in enumerate(st_aligned)
    if data['classif_from_GUI']['Classification'][i] not in EXCLUDE_UNITS
]

clean_cluster_ids = [
    cluster_id for i, cluster_id in enumerate(cluster_ids)
    if data['classif_from_GUI']['Classification'][i] not in EXCLUDE_UNITS
]

# Dealing with delay by subtracting 50ms from all units in clean_st_aligned.
clean_st_aligned = [(unit - NP_DELAY) for unit in clean_st_aligned]

########### 3. Calculate FR ###########
# Calculating spike count per neuron within every frame for the stimulus
print(f"---- Calculating firing rates for stimulus ({stimulus_name}) in experiment ({experiment_date}) ")
firing_rates = []
for ttl in stimulus_ttls:
    # We have 192 frames starting the firing of the ttls
    # 192 frames -> 1.6 seconds
    # Timespan of 192 frames = 1.6 * 30_000 (NP_FRQ) = 48_000
    # Timespan of 1 frame = 250
    # We calculate average firing rate per frame (250)
    orientations_frs = []
    for i in range(NUM_FRAMES):
        lower_bound = ttl + (i * FRAME_SPAN)
        upper_bound = ttl + ((i+1) * FRAME_SPAN)
        # print("lower_bound: ", lower_bound)
        # print("upper_bound: ", upper_bound)
        # print("\n")
        neuron_frs = []
        for neuron in clean_st_aligned:
            spike_count = np.where((neuron >= lower_bound) & (neuron < upper_bound), True, False).sum()
            # Duration: diff_in_ttl / 30000
            # Rate: (sum/duration)
            firing_rate = (spike_count * NP_FRQ) / (upper_bound - lower_bound)
            # if spike_count > 0:
                # print("spike_count: ", spike_count)
                # print("firing_rate: ", firing_rate)
                # print("(upper_bound - lower_bound): ", (upper_bound - lower_bound))
                # print("\n")
            neuron_frs.append(firing_rate)
        orientations_frs.append(neuron_frs)
    firing_rates.append(orientations_frs)

# Shape (stimulus_length, NUM_FRAMES, num_of_units)
fr_per_stimulus = np.array(firing_rates)

########### 4. Loading the stimulus ###########
# Putting the frames and the firing rates together (Need to match type of stimulus chosen)
stim_params = np.load(stim_params_file, allow_pickle=True, encoding='latin1').item()
orientations = stim_params['stimulus']['sequence']['orientations']

# Append all stimuli files together
stimulus = []
for orientation in orientations:
  stim_file = f"moving_bar_{orientation}.npy"
  stim_path = os.path.join(PSYCHOPY_STIMULI, stim_file)
  stimulus.append(np.load(stim_path))

# Shape (stimulus_length, NUM_FRAMES, HEIGHT, WIDTH)
stimulus = np.array(stimulus)

########### 5. Reshape and Save ###########
# (stimulus_length * NUM_FRAMES, num_of_units)
fr_per_stimulus = fr_per_stimulus.reshape((stimulus_length * NUM_FRAMES, fr_per_stimulus.shape[-1]))

# (stimulus_length * NUM_FRAMES, WIDTH, HEIGHT)
stimulus = stimulus.reshape((stimulus_length * NUM_FRAMES, stimulus.shape[-1], stimulus.shape[-2]))

print("Firing Rates Shape: ", fr_per_stimulus)
print("Stimulus Shape: ", stimulus)

# Save the data together (4 -> moving_bar, 5 -> moving_grating)
save_stim_file_name = "{}_4_stimulus_moving_bar.npy".format(experiment_tag)
save_fr_file_name = "{}_4_fr_moving_bar.npy".format(experiment_tag)
save_cluster_ids_file_name = "{}_cluster_ids.npy".format(experiment_tag)

with open(os.path.join(TRAINING_DATA_SAVE_DIR, save_stim_file_name), 'wb') as f:
    np.save(f, stimulus)

with open(os.path.join(TRAINING_DATA_SAVE_DIR, save_fr_file_name), 'wb') as f:
    np.save(f, fr_per_stimulus)

with open(os.path.join(TRAINING_DATA_SAVE_DIR, save_cluster_ids_file_name), 'wb') as f:
    np.save(f, np.array(clean_cluster_ids))

print("Created ", save_stim_file_name)
print("Created ", save_fr_file_name)
print("Created ", save_cluster_ids_file_name)


########### 6. Plot for validation ###########
if VALIDATION_PLOT:
    import matplotlib.pyplot as plt

    # firing_rates shape: (T, N)
    T, N = fr_per_stimulus.shape
    units = np.random.choice(N, 3, replace=False)

    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    for ax, u in zip(axs, units):
        ax.plot(fr_per_stimulus[:, u])
        ax.set_title(f"Unit {u}")
        ax.set_ylabel("Firing rate")

    axs[-1].set_xlabel("Time")

    plt.tight_layout()
    plt.show()
