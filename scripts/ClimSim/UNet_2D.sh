#!/bin/bash

module load anaconda/2020.11
module load cuda/11.8
source activate climsim

model_name=UNet_2D

python -u run.py \
  --task_name 2d \
  --is_training 1 \
  --root_path ./dataset/ClimSim_2D \
  --model_id 2d \
  --model $model_name \
  --data ClimSim2D \
  --in_channel 25 \
  --out_channel 14 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 300 \
  --batch_size 156 \
  --patience 100 \
  --learning_rate 0.0001 \
  --lradj cosine \
  --inverse \
  --postprocess