#!/bin/bash

module load anaconda/2020.11
module load cuda/11.8
source activate climsim

model_name=Transformer_ed

python -u run.py \
  --task_name ed \
  --is_training 0 \
  --root_path ./dataset/ClimSim \
  --model_id ed \
  --model $model_name \
  --data ClimSim1D \
  --in_channel 25 \
  --out_channel 14 \
  --dropout 0.2 \
  --e_layers 6 \
  --d_layers 6 \
  --d_model 512 \
  --d_ff 2048 \
  --nhead 8 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10 \
  --batch_size 1600 \
  --patience 3 \
  --learning_rate 0.0001 \
  --lradj fix \
  --inverse \
  --use_multi_gpu \
  --devices 0,1 \
  --sample_rate 0.31 \
  --weight_decay 0