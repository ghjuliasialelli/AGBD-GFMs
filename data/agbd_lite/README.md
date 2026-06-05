# AGBD-Lite

ℹ️ This repository contains the scripts used to generate the AGBD-Lite dataset, a subset of the [AGBD dataset](https://huggingface.co/datasets/prs-eth/AGBD). 


### 🧬 Structure
|_ `/eval` : code to evaluate the `nico_net` model trained from scratch on the AGBD-Lite dataset.  
|_ `/helper` : code to explore the data.   
|_ `/indices` : folder containing the indices that were selected for sub-sampling from the AGBD dataset, for each region and year.   
|_ `/vis` : folder containing plots comparing the pre- and post- subsampling distribution of the AGB and biome values, for each region.  
|_ `compute_statistics.py` : script that computes the statistics on the AGBD-Lite train dataset, `AGBD-Lite-statistics.pkl`.  
|_ `exploration.ipynb` : notebook to generate the `regional_distributions.pkl` and `tile_per_region.pkl` files.  
|_ `features.json` : describes the features we want to keep in the AGBD-Lite dataset.   
|_ `find_indices.py` : script that selectes a subset of indices of the AGBD Dataset.  
|_ `gen_subsample.py` : script that actually performs the subsampling of the AGBD Dataset into an AGBD-Lite version.

And accompanying files:
* `regional_distributions.pkl` : file containing the reference distributions (in the AGBD dataset) for all regions.
* `tile_to_file.pkl` : file containing a mapping from each S2 tile to the AGBD `.h5` file it is contained in.
* `tile_to_region.pkl` : file containing a mapping from S2 tiles to regions, represented as numbers, ordered alphabetically.
* `tiles_per_region.pkl` : file containing a list of S2 tiles for each region.


### ⬇️ Accessing the data
You can download the necessary files on Zenodo. They include:
* `AGBD-Lite_<mode>.h5` with `mode` $\in \{train, test, val \}$ : the AGBD-Lite dataset.
* `AGBD-Lite-statistics.pkl` : the statistics computed on `AGBD-Lite_train.h5`.
* `supervised_test_results.h5` : a file containing the predictions of our supervised model on the test dataset.
* `/supervised_weights` : a folder containing the trained weights of supervised model.
