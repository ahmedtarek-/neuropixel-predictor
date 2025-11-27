# Stimuli Script
This folder contains scripts that are used to generate the actual stimulus used for the experiment.
**It's mainly used for sanity checks.**

The stimuli matrices are generated in `gen_stimuli`

- Note that you should use a standalone environment with python 3.10 or 3.8 to be able
to install psychopy (according to https://www.psychopy.org/download.html)

    ```bash
    conda create -n "psychopy" python=3.10
    conda activate psychopy

    conda install conda-forge::numpy
    pip install psychopy
    ```
