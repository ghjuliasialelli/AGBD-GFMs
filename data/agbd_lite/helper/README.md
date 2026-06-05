## Using the AGBD-Lite dataset

### 🧬 Structure

|_ `/results` : folder containing the `.h5` files with the results of the models on the AGBD-Lite test dataset. It contains `supervised_test_results.h5`, the predictions of our supervised baseline on the test dataset.  
|_ `config.yaml` : the config file to load the `GEDIDataset()`.  
|_ `dataset.py` : file defining the `GEDIDataset()` class.  
|_ `embeddings_train.csv` : embeddings used in the `GEDIDataset()`.  
|_ `helper_functions.py` : helper functions used by the `GEDIDataset()`.  
|_ `overview.ipynb` : example on how to iterate over the `GEDIDataset()`.  
|_ `plot.py` : example code to compare the performance of a model vs. the supervised baseline on the AGBD-Lite test set.

**TL;DR** — Take a look at the `overview.ipynb` file to learn how to load the data.


### ⬇️ Accessing the data
You can download the necessary files on Zenodo. They include:
* `AGBD-Lite_<mode>.h5` with `mode` $\in \{train, test, val \}$ : the AGBD-Lite dataset.
* `AGBD-Lite-statistics.pkl` : the statistics computed on `AGBD-Lite_train.h5`.
* `baseline_<i>.ckpt` : the weights of the ensemble of baseline supervised models, with $i \in [1,2,3]$.


### ⚙️ Building the environment
To install the packages required to run this code, you can simply run the following commands, which will create a conda virtual environment called agbdlite. For more details, follow the instructions on [pytorch.org](https://pytorch.org/get-started/locally/).
```
conda create -n agbdlite python=3.11 pytorch cudatoolkit numpy h5py pandas scipy omegaconf matplotlib -c pytorch -c nvidia -c conda-forge

conda activate agbdlite
```