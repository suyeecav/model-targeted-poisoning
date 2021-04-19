# About
This folder of maintains code for comparison experiments from Appendix E

# Install Dependency
You'll need to set up a fork of the data-poisoning-release repository for the `Koh & Liang (2017)` attack from [here](https://github.com/iamgroot42/data-poisoning-release)
```
git clone --recurse-submodules git@github.com:iamgroot42/data-poisoning-release.git
```

# Run the attacks

1. `Biggio et. al. (2011)` : Implemented in `biggio_attack.py`. 
2. `Koh & Liang (2017)` : Implemented in `data-poisoning-release/koh_liang_attack.py`, needs to be run with Python 2 (tested with 2.7.18) from inside that folder.
3. `Demontis et. al. (2019)` : Implemented in `demontis_attack.py`.

# Useful files

1. `evaluate_secml.py`: Computing performance of classifiers using ratios of poisoned data available, specific for secML-based models.
2. `train_with_poison.py`: Similar funcion as above: reads data from directory where poisoned data as generated sequentially, then trains models with certain ratios of that poisoned data.

# Analyzing trends

Poisoned data is dumped by all attacks (check each file to se which folder), which can then be used for training models with that poisoned data, performing analyses, etc.
