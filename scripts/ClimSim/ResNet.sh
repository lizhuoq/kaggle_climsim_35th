#!/bin/bash

module load anaconda/2020.11
module load cuda/11.8
source activate climsim

model_name=ResNet

python -u run.py \
  --task_name temp \
  --is_training 1 \
  --root_path ./dataset/ClimSim \
  --model_id temp \
  --model $model_name \
  --data ClimSim1D \
  --in_channel 25 \
  --out_channel 368 \
  --n_layers 50 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 100 \
  --batch_size 16000 \
  --patience 3 \
  --learning_rate 0.001 \
  --lradj cosine \
  --inverse \
  --use_multi_gpu \
  --devices 0,1 \
  --sample_rate 0.31 \
  --postprocess