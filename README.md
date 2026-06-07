# neuropixel-predictor


## A. Working with environment
- To create a new conda env using
```bash
conda env create -f environment.yml
```

- To update environment with new packages
```bash
conda env update --file environment.yml  --prune
```

- Need to have Weight&Biases account to track models. Then run

```bash
# You should have API key to enter at this point
wandb login
```



## B. Working with data
Firing Rate data is extracted from `data/data_single_unit/2023-03-15_11-05-00_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.pkl`.

The available data is.

```python
['mb', 'Sl36x22_d_3', 'Sd36x22_l_3', 'chirp', 'csd', 'Nat_Mov', 'Nat_Mov_sw',
'Nat_Mov_sc', 'Sl36x22_d_3_2', 'Sd36x22_l_3_2', 'mb_2', 'Nat_Mov_2', 'Nat_Mov_sw_2',
'Nat_Mov_sc_2', 'csd_2', 'chirp_2', 'cm_18x11_2', 'Sl36x22_d_3_3', 'Sd36x22_l_3_3',
'mb_3', 'Nat_Mov_3', 'Nat_Mov_sw_3', 'Nat_Mov_sc_3', 'csd_4', 'chirp_3', 'cm_18x11_3',
'cm_18x11_1', 'TTLs_FM_start', 'TTLs_FM_stop', 'TTLs_act_start', 'TTLs_act_stop']
```

The current data that was extracted, incorporated and sanity checked:

- [x] `Sl36x22_d`: Sparse Noise Light (on dark background) 
- [x] `Sd36x22_l`: Sparse Noise Dark  (on light background)
- [x] `csd`: Current Source Density - Checker Board (binary)
- [ ] `cm_18x11`: Checkermap (with greys)
- [ ] `mb`: Moving Bar *
- [ ] `mg_sin`: Moving Grating *
- [ ] `mg_sq`: Moving Grating *
- [ ] `chirp` *
- [ ] `lo`: Looming (not in all recordings)
- [ ] `sg`: Static Gratings (not in all recordings)

- [ ] `Nat_Mov` will check later


The Scripts README details which scripts have been used to generate the stimului matrices
that have been used in training.

**Data Sanity**

Use the `run_stimulus` script to generate visual stimuli and compare them against the matrices
that will be used as input to the network. This should be done to every stimulus mentiond
above.


## C. Exact Steps to Run Pipeline

Let's say we have this list of stimuli

```
2022-12-20_15-08-10
processing with mb having 120 TTLs
processing with Sl36x22_d_3 having 4561 TTLs
processing with Sd36x22_l_3 having 4561 TTLs
processing with mg_sq having 240 TTLs

============
2022-12-21_13-09-10
processing with mb having 120 TTLs
processing with Sl36x22_d_3 having 4561 TTLs
processing with Sd36x22_l_3 having 4561 TTLs
processing with mg_sq having 240 TTLs
```

We need these exact steps for each experiment (i.e twice in that case)

1. Generate the desired stimuli sizes using psychopy (check scripts section)

2. Calculate Firing Rate (edit script for each stimuli with correct name and paths)
    1. Run `calculate_firing_rate.py` once for 'Sl36x22_d_3'
    2. Run `calculate_firing_rate.py` once for 'Sd36x22_d_3'
    3. Run `calculate_firing_rate_mb.py` once for 'mb'
    4. Run `calculate_firing_rate_mb.py` once for 'mg_sq'

3. Generate Training data
    1. Choose the stimuli [sl, sd, mb, mg_sq] from the end of `gen_training_and_test_data.py`
    2. Choose the desired experiment
    3. Choose where to save the training files (ex. `path/TRAINING_FOLDER`)
    4. Run the `gen_training_and_test_data.py` script

4. Go to simplified_notebooks, paste the `path/TRAINING_FOLDER`, run the notebook
