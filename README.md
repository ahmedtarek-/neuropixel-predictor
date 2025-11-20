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

The current data that was extracted and incorporated into the training are:

- [] `Sd36x22_l_3`: Sparse Noise Light (on dark background) 
- [] `Sl36x22_d_3`: Sparse Noise Dark  (on light background)
- [] `csd`: Checkermap
- [] `mb`: Moving Bar
- [] `chirp`
- [] `Nat_Mov`

The Scripts README details which scripts have been used to generate the stimului matrices
that have been used in training.


