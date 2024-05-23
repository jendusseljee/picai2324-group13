#!/bin/bash

module load 2023
module load py

# echo "Installing dependencies"
# python -m pip install --user --upgrade pip
# python -m pip install --user scikit-build
# python -m pip install --user -r $HOME/picai_2324/src/picai_baseline/unet/requirements.txt

echo "Copying data to scratch..."
cp -r $HOME/picai_2324/workdir $TMPDIR/workdir
cp -r $HOME/picai_2324/output $TMPDIR/output
echo "Done copying!"
 


export PYTHONPATH=$PYTHONPATH:$HOME/picai_2324/src

echo "Inferring unet"
python infer.py \
  --task-dir $TMPDIR/workdir/nnUNet_raw_data/Task2203_picai_baseline/ \
  --weights-dir $TMPDIR/output/UNet/weights/ \
  --folds 0 1 2 3 4 \
  --output-dir $TMPDIR/output/UNet/infer/
echo "Done!"


echo "Copying data back to home"
mkdir -p $HOME/picai_2324/output/UNet/
cp -r $TMPDIR/output/UNet/infer/ $HOME/picai_2324/output/UNet/infer/
