#!/bin/bash

#Set job requirements
#SBATCH -J train
#SBATCH --partition gpu
#SBATCH -N 1
#SBATCH -t 120:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --gpus 1
#SBATCH --mem 40G
#SBATCH --output slurm-logs/train-%j.out

module load 2023
module load py

echo "Installing dependencies"
python -m pip install --user --upgrade pip
python -m pip install --user scikit-build
python -m pip install --user -r $HOME/picai_2324/src/picai_baseline/unet/requirements.txt

echo "Copying data to scratch..."
cp -r $HOME/picai_2324/workdir $TMPDIR/workdir
echo "Done copying!"
 
echo "Replacing paths in overviews..."
python replace_paths_in_overview.py \
  $TMPDIR/workdir/results/UNet/overviews/Task2203_picai_baseline
  $HOME/picai_2324
  $TMPDIR
echo "Done replacing paths!"

export PYTHONPATH=$PYTHONPATH:$HOME/picai_2324/src

echo "Training unet"
python -u src/picai_baseline/unet/train.py \
  --weights_dir=$TMPDIR/workdir/results/UNet/weights/ \
  --overviews_dir=$TMPDIR/workdir/results/UNet/overviews/Task2203_picai_baseline \
  --folds 0 1 2 3 4 --max_threads 6 --enable_da 1 --num_epochs 250 \
  --validate_n_epochs 1 --validate_min_epoch 0
echo "Done!"


echo "Copying data back to home"
cp -r $TMPDIR/workdir/results/UNet/weights/ $HOME/picai_2324/output/UNet/weights/
