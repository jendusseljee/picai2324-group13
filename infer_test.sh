#/bin/bash

module load 2023
module load py

python infer.py \
  --task-dir /home/dthijs/picai_2324/workdir/nnUNet_raw_data/Task2203_picai_baseline/ \
  --weights-dir /home/dthijs/picai_2324/output/UNet/weights/ \
  --folds 0 1 2 3 4 \
  --output-dir $TMPDIR/output/UNet/infer/