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

- [] `Sl36x22_d`: Sparse Noise Light (on dark background) 
- [] `Sd36x22_l`: Sparse Noise Dark  (on light background)
- [] `csd`: Current Source Density - Checker Board (binary)
- [] `cm_18x11`: Checkermap (with greys)
- [] `mb`: Moving Bar
- [] `mg_sin`: Moving Grating
- [] `mg_sq`: Moving Grating
- [] `chirp`
- [] `lo`: Looming
- [] `sg`: Static Gratings

- [] `Nat_Mov` will check later


The Scripts README details which scripts have been used to generate the stimului matrices
that have been used in training.

**Data Sanity**

Use the `run_stimulus` script to generate visual stimuli and compare them against the matrices
that will be used as input to the network. This should be done to every stimulus mentiond
above.


