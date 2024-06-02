This repository contains the code of group 13 for the course AI in Medical Imaging 2024 at Radboud University.
The goal for the project was to improve the performance on the [PI-CAI challenge](https://pi-cai.grand-challenge.org/).

The code was based on the code [this repository](https://github.com/DIAGNijmegen/picai_baseline), which was used as the starter template.

## Overview
- The src folder contains the source code we used for the deep learning models based on the baseline source code. The architectures we have used are in src/picai_baseline/unet/training_setup/neural_networks. Additional changes were made to src/picai_baseline/unet/training_setup/data_generator.py to support the use of tabular data during the training process. Some minor changes have been made to other files to support loading our new model architectures.
- The folder data_analysis contains notebooks and csv files that were used to explore the tabular data we have and impute missing values in the clinical data.
- The folder outputs contains the code we used to analyse the performance of all of our models, as well as the code for our xgboost model.
- Lastly, the folder grand_challenge_algorithms contains all the docker projects used for making submissions to Grand Challenge. All of these projects were based on the [provided baselines](https://github.com/DIAGNijmegen/picai_unet_semi_supervised_gc_algorithm).