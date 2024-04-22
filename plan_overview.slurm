#!/bin/bash

#Set job requirements
#SBATCH -J plan_overview
#SBATCH -N 1
#SBATCH -t 24:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem=16G
#SBATCH --output=slurm-logs/overview-%j.out

module load 2023
module load py

export PYTHONPATH=$PYTHONPATH:$HOME/picai_2324/src
 
echo "Creating overview..."
python src/picai_baseline/unet/plan_overview.py --task=Task2203_picai_baseline --workdir=workdir --preprocessed_data_path=$HOME/picai_2324/workdir/nnUNet_raw_data/{task} --overviews_path=results/UNet/overviews/{task} --splits=picai_pub
echo "Done!"
