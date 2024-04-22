#!/bin/bash

#Set job requirements
#SBATCH -J preprocess
#SBATCH -N 1
#SBATCH -t 24:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem=16G
#SBATCH --output=slurm-logs/%j.out

module load 2023
module load py
 
echo "Preparing data..."
python src/picai_baseline/prepare_data_semi_supervised.py --spacing 3.0 0.5 0.5 --matrix_size 20 256 256 --inputdir input --workdir workdir
echo "Done!"
