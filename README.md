# AQA
project for AQA at Tu Delft 2025

Paper this program is based on:
https://doi.org/10.1103/PhysRevResearch.5.013211

## How to install

1. Create a python virtual environment using `python -m venv .`
2. Activate the virtual environment
3. Install the dependencies using `pip install -r requirements.txt`

## What do the files do

### variable_layers.py
This program generates variable layers for the supplied M values in the for loop `for m in [...]:`.
It generates the layers up to `L_max` and picks the top `K` circuits to use as a base for the next layer.

### fixed_layers.py
This program generates fixed layers optimized using the supplied M values in the for loop `for m in [...]:`.
It generates the layers up to `L_max` and picks the top `K` circuits to use as a base for the next layer.

### plotting.ipynb
This notebook contains code to plot the output `data.json` from the above.

### image_samples.py
This program creates and then displays a circuit containing the different layers used in `fixed_layers.py`.

### kernel_selection.ipynb
This notebook contains the code for the plots on comparing different kernel selection metrics.
