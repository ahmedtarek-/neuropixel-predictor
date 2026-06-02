"""
For moving bar and moving grating, calculates firing rate.

Removes the units that are 'Flags' or 'Dendrites' (based on manual classification)
{'Flag', 'MPW-Dendrite', 'SU-Small', 'SU-Positive'}

Things that needs to be changed:
    1. 'experiment_names'       -> Single unit files.
    2. 'STIMULUS_NAME':         -> Which stimulus to calculate FR for.
    3. 'PSYCHOPY_STIMULI'       -> Where to load stimuli from
    4. 'STIM_RESP_SAVE_DIR'     -> Where to save stimuli-responses pair data
    
Hints:
    - `data["stim_params_files"]["mb"]["stimulus"]["sequence"]["orientations"]`
    should match number of TTls in that protocol.

    - `data["stim_params_files"]["csd"]["stimulus"]["trials"]` gives number
    of trials in the checkerboard.


Time bins Normalization (VERY IMPORTANT):
    - In `calculate_firing_rate.py` script, we calculate the firing rate of sparse noise darks
    and use that as references for what rough frequency we apply for other binning.
    - Using the following snippet gives us mean difference in ttls that was used to calculate
    firing rate.
    ```
    sd_ttls = data['events']['Sd36x22_l_3']
    np.diff(sd_ttls).mean()
    ``` 
    - The mean difference in ttls of sparse noise is 3004 which means 100ms
    - In this script the firing rate is calculated per 250 (in 30khz), which means
        8.3ms.
    - Therefore the normalization factor is 8.3/100


@author: Ahmed Abdalfatah (@ahmedtarek-)
"""

import os
import re
import numpy as np
import pandas as pd


########### A. Define variables and data location ###########
DATA_FOLDER = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data'

PSYCHOPY_STIMULI = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli/Psychopy-64x36' # Psychopy-36x22
STIM_RESP_SAVE_DIR = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli-Responses-64-36'

SINGLE_UNIT_FOLDER = os.path.join(DATA_FOLDER, 'data-single-unit')
STIM_PARAMS_FOLDER = os.path.join(DATA_FOLDER, 'Stimuli-Params')

MLI_THRESHOLD = 0.5 # Include only units with more thab 50% modulation index.
MLI_FILE = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/MLI_MB/storage_MLI_sparse_noise.npy'

NP_FRQ = 30_000
NP_DELAY = (50 * NP_FRQ) / 1000

NORMALIZATION_FACTOR = 8.3/100

FRAME_SPAN = (1/120) * NP_FRQ

EXCLUDE_UNITS = ['Flag', 'MPW-Dendrite', 'SU-Small', 'SU-Positive']

######## B. Choosing stimuli  ########
# Choose stimulus (ex. mb, mg_sq)
STIMULUS_NAME = 'mg_sq'
NUM_FRAMES = 192 if STIMULUS_NAME == 'mb' else 180

# Define stimuli_params_key
mb_stimuli_params_keys = {
    '2022-12-20_15-08': '20221220/c11emb_params.npy',
    '2023-03-15_11-05': '20230315bis/c11cmb_params.npy',
    '2023-03-15_15-23': '20230315/c11cmb_params.npy',
    '2022-12-21_13-09': '20221221/b11emb_params.npy',
    '2023-02-23_08-57': '20230223/d11cmb_params.npy',
    '2023-03-16_12-16': '20230316/c11cmb_params.npy',
    '2023-03-21_16-17': '20230321/b11cmb_params.npy',
    '2023-03-22_12-22': '20230322/b11cmb_params.npy',
    '2023-04-13_12-35': '20230413/b11dmb_params.npy',
    '2023-04-14_11-48': '20230414/b11cmb_params.npy',
    '2023-04-17_12-26': '20230417/b11dmb_params.npy',
    '2023-04-18_12-10': '20230418/b11dmb_params.npy',
}

mg_stimuli_params_keys = {
    '2023-03-21_16-17': '20230321/a11hmg_sq_params.npy',
    '2023-03-22_12-22': '20230322/a11hmg_sq_params.npy',
    '2023-04-13_12-35': '20230413/a11hmg_sq_params.npy',
    '2023-04-14_11-48': '20230414/b11gmg_sq_params.npy'
}

######## C. Defining helper method for modulation index based filtering ########
class StimulusParamFileNotFound(Exception):
    pass
class MLINotFound(Exception):
    pass

def fetch_mli_good_indices(experiment_name):
    """
    Given an experiment, returns the top MLI_THRESHOLD based on
    modulation index of sparse noise dark.
    """
    mli_file = np.load(MLI_FILE, allow_pickle=True, encoding='latin1').item()

    if experiment_name not in mli_file.keys():
        raise MLINotFound("No MLI entry found for this experiment")

    mli = mli_file[experiment_name]['Sd36x22_l_3']['local_MLI'] # Sd36x22_l_3_3
    abs_mli = np.abs(mli)
    good_indices = np.where(abs_mli > MLI_THRESHOLD) # Cut lowest MLI_THRESHOLD and cast as int

    return set(good_indices[0])

######## D. Defining main firing rates calculation ########
def calculate_firing_rate_mb(experiment_name, validation_plot=False):
    ######## 1. Extracting experiment date and reading data ########

    # Defining pattern to extract right date (notice that the capture group finishes before the last two digits)
    pattern = '^([0-9\\-]+\\_[0-9]{2}\\-[0-9]{2})\\-[0-9]{2}\\_Complete.+$'
    experiment_date = re.search(pattern, experiment_name).groups()[0]
    
    print("\n=====================")
    print("Processing Experiment -> exp_date: ", experiment_date)

    try:
        if STIMULUS_NAME == 'mb':
            stim_params_file = mb_stimuli_params_keys[experiment_date]
        elif STIMULUS_NAME == 'mg_sq':
            stim_params_file = mg_stimuli_params_keys[experiment_date]
    except KeyError as e:
        raise StimulusParamFileNotFound(
            "Couldn't find stimulus parameter file for {}, will skip!".format(experiment_date)
        )
    single_unit_file = os.path.join(SINGLE_UNIT_FOLDER, experiment_name)
    stim_params_file = os.path.join(STIM_PARAMS_FOLDER, stim_params_file)

    ######## D. Main firing rate calculation ########
    # Load pickle
    data = pd.read_pickle(single_unit_file)

    stim_params = np.load("/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli-Params/20230322/a11hmg_sq_params.npy", allow_pickle=True, encoding='latin1').item()

    # Reshape because of weird shape
    data = data.reshape((1))[0]

    # To get which stimulius are there
    # data['events'].keys()
    stimulus_ttls = data['events'][STIMULUS_NAME]

    # Stimulus Length
    stimulus_length = len(stimulus_ttls)

    # To get spiketimes aligned
    st_aligned = data['spiketimes_aligned']

    # To get the cluster ids (unique and important to track)
    cluster_ids = data['classif_from_GUI']['clusterIds']

    # Modulation index good indices
    try:
        mli_good_indices = fetch_mli_good_indices(experiment_name.split('_Complete')[0])
    except MLINotFound as e:
        # Set mli_good_indices to all indices
        print(e)
        mli_good_indices = np.arange(len(st_aligned))

    print("Number of units before cleaning: ", len(st_aligned))
    print("Number of units after MLI cleaning: ", len(mli_good_indices))

    ######## 2. Cleaning Data ########

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


    # Dealing with delay by subtracting 50ms from all units in clean_st_aligned.
    clean_st_aligned = [(unit - NP_DELAY) for unit in clean_st_aligned]

    ########### 3. Calculate FR ###########
    # Calculating spike count per neuron within every frame for the stimulus
    print(f"---- Calculating firing rates for stimulus ({STIMULUS_NAME}) in experiment ({experiment_date}) ")
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
                firing_rate = firing_rate * NORMALIZATION_FACTOR
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
    suffix = 'bar' if STIMULUS_NAME == 'mb' else 'grating_square'
    for orientation in orientations:
      stim_file = f"moving_{suffix}_{orientation}.npy"
      stim_path = os.path.join(PSYCHOPY_STIMULI, stim_file)
      stimulus.append(np.load(stim_path))

    # Shape (stimulus_length, NUM_FRAMES, HEIGHT, WIDTH)
    stimulus = np.array(stimulus)

    ########### 5. Reshape ###########
    # (stimulus_length * NUM_FRAMES, num_of_units)
    fr_per_stimulus = fr_per_stimulus.reshape((stimulus_length * NUM_FRAMES, fr_per_stimulus.shape[-1]))

    # (stimulus_length * NUM_FRAMES, WIDTH, HEIGHT)
    print("Stimulus shape: ", stimulus.shape)
    stimulus = stimulus.reshape((stimulus_length * NUM_FRAMES, stimulus.shape[-2], stimulus.shape[-1]))

    ########### 6. Remove empty stimulus frames ###########
    # Indices of frames with only zero
    non_zero_indices = np.count_nonzero(stimulus, axis=(1,2)) != 0

    zero_frames_count = stimulus.shape[0] - non_zero_indices.sum()
    print("\nDetected {} empty frames (out of {}). Will remove them".format(zero_frames_count, stimulus.shape[0]))

    stimulus = stimulus[non_zero_indices]
    fr_per_stimulus = fr_per_stimulus[non_zero_indices]

    ########### 7. Center around 0 [-1, 1] ###########
    # Check if stimulus is not normalized
    if stimulus.mean() > 1:
        print("\n[Vorsicht] Stimulus is not centered around 0.")
        print("---- Assuming the stimulus has range [0,255]")
        print("---- Will normalize to [-1,1]")
        stimulus = (stimulus / 127.5) - 1

    print("Firing Rates Shape: ", fr_per_stimulus.shape)
    print("Stimulus Shape: ", stimulus.shape)

    ########### 8. Save stimulus-response pairs ###########
    # Save the data together (4 -> moving_bar, 5 -> moving_grating)
    suffix = 'bar' if STIMULUS_NAME == 'mb' else 'grating'
    numb = 4 if STIMULUS_NAME == 'mb' else 5

    save_stim_file_name = "{}_{}_stimulus_moving_{}.npy".format(experiment_date, numb, suffix)
    save_fr_file_name = "{}_{}_fr_moving_{}.npy".format(experiment_date, numb, suffix)
    save_cluster_ids_file_name = "{}_cluster_ids.npy".format(experiment_date)

    with open(os.path.join(STIM_RESP_SAVE_DIR, save_stim_file_name), 'wb') as f:
        np.save(f, stimulus)

    with open(os.path.join(STIM_RESP_SAVE_DIR, save_fr_file_name), 'wb') as f:
        np.save(f, fr_per_stimulus)

    with open(os.path.join(STIM_RESP_SAVE_DIR, save_cluster_ids_file_name), 'wb') as f:
        np.save(f, np.array(clean_cluster_ids))

    print("Created ", save_stim_file_name)
    print("Created ", save_fr_file_name)
    print("Created ", save_cluster_ids_file_name)


    ########### 6. Plot for validation ###########
    if validation_plot:
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

experiment_names = [
    # "2023-01-27_12-58-44_Complete_spiketime_Header_TTLs_withdrops_n_withGUIclassif.pkl",
    # "2022-12-20_15-08-10_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl",
    # "2023-03-15_11-05-00_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl",
    # "2023-03-15_15-23-14_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl",
    # "2022-12-21_13-09-10_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl",
    # "2023-02-23_08-57-20_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl",
    # "2023-03-16_12-16-07_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl",
    "2023-03-21_16-17-18_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl",
    "2023-03-22_12-22-12_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl",
    "2023-04-13_12-35-02_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl",
    "2023-04-14_11-48-04_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl",
    # "2023-04-17_12-26-07_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl",
    # "2023-04-18_12-10-34_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl",
]


for experiment_name in experiment_names:
    try:
        calculate_firing_rate_mb(experiment_name)
    except StimulusParamFileNotFound as e:
        print(e)
        continue
