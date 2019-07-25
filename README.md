# Installation

1. Download this git repository.  
        `Note: This project is developed for Python3, Python2 is not supported.`
2. Install the python requirements (preferably in a virtual environment or similar) using  
        `pip install -r requirements.txt`.  
   The default virtual environment is named `contiki_env` and located in the home directory.
   If you want to use a different name or location some pathes might need adaption (e.g. the ones in `create_all.py`).  
3. This project is based on the RPL Attacks Framework (https://github.com/dhondta/rpl-attacks) commit `0e561d0e4edcd989c2e715b6a884aa47b0cf6022`.
    Follow the Normal Installation steps of this version. They can be found in `RPL_Attacks_Framework_README.md`.
    The download of the git should be skipped since all data is included in this git repository.
4. Ensure that Python finds this repository. This can be done temporarily by the export command:  
    `export PYTHONPATH="$PYTHONPATH:path/to/this/repository"`  
    `Note: The export command has to be called again after a reboot.`
    
# User Guide

This project assumes the following project structure:

parent directory (e.g. home directory)  
|  
|--RPL-Attacks-Data (directory containing simulation data)  
|  
|--rpl-attacks (this git repository, contains source code and models)  
If you prefer another structure, you will have to adapt some paths in the code.

## RPL Attacks data

The data used for generating the models can be found in the `RPL-Attacks-Data` repository.
Note, that the data in this repository has been used for training, so if you compare the scores to the ones in the
Python implementation they will differ significantly.

## Data Generation

The RPL-Attacks-Data git repository contains the data used in the paper.

In order to generate new data:
1. Generate a new directory in RPL-Attacks-Data `<new dir>` (e.g. 21l).
2. Generate a network architecture configuration and store it as `<new dir>/<new dir>.json`.  
    `Note: Example configurations can be found in the existing data sets.`
3. Configure the new data set in `create_all.py`. Also ensure the virtual environment and the paths are correct.
4. Run the create_all script using `python create_all.py`.
If you encounter problems, ensure you are in the correct python environment.
It might happen, that the simulation fails and you receive an out-of-bounds error.
In this case you most likely selected more than five models while the current version only supports five models due to memory constraints.
We advise to use the models for 1-5 neighbors. Larger networks might require a different selection.
In order to mark a model to not use the for the C-Implementation, you have to rename it from `<number neighbors>.pkl` to something
different, preferably `<number neighbors>.back`. The model can be found in the `global_train` directory.
5. Wait until the simulation is over. This might take several hours, larger simulations can take several days.
6. If you did not get any errors or exceptions during the simulation, you should have your data sets in the specified directories.

Note: If you want to calculate the size statistics, you might want to create an additional data set in 
`RPL-Attacks-Data/without_AD/1l` where you disable anomaly detection. This can be done by commenting out
`CFLAGS += -DACTIVATE_AD` in `templates/experiment/motes/Makefile`.

## Training models and calculating statistics

The main files for training and statistic calculation are `pretraining/main.py` and `pretraining/config/batch_config.py`.
`main.py` just simply has to be run using Python interpreter with the installed packages.
`batch_config.py` is used to configure the project. Here you can specify the data sets, the attacks, and which modules to run.

The results of the following modules will be stored in the directory of the respective data set and attack.

- `extract_features`: Extracts the features measured in the simulation into csv files so other modules can easily access them.
    Once run on a data set, this module does not have to be run again.
- `draw_udp_graphs`: Draws simple graphs illustrating the UDP flow in the simulation.

The next modules generate the models, calculate the threshold, and evaluate some statistics.
The results are stored in the `global_train` directory.

- `run_generate_data_model`: Generates the KDE models based on the features of the selected data sets.
    Afterwards the KDE models will be approximated using cubic splines. This will overwrite older models without warning.
- `run_calculate_threshold`: Calculates the threshold given the percentage of false positives you want.
- `run_statistics`: Calculates the true positives and false positives as seen in the paper.
- `run_power_statistics`: Extracts the number of ticks each of other functions requires and calculates some statistics.
- `run_size_statistics `: Uses the size command to check the size of contiki with and without anomaly detection.
