#!/bin/bash

module load anaconda/2020.11
module load cuda/11.8
source activate climsim

model_name=UNet

python -u run.py \
  --task_name 1d \
  --is_training -2 \
  --root_path ./dataset/ClimSim \
  --model_id 1d \
  --model $model_name \
  --data ClimSim1D \
  --in_channel 25 \
  --out_channel 14 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 50 \
  --batch_size 16000 \
  --patience 3 \
  --learning_rate 0.001 \
  --lradj cosine \
  --inverse \
  --use_multi_gpu \
  --devices 0,1 \
  --postprocess 