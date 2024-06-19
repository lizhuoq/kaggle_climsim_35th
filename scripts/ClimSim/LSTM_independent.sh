#!/bin/bash

module load anaconda/2020.11
module load cuda/11.8
source activate climsim

model_name=LSTM_independent

# 7 * 256
python -u run.py \
  --task_name 1d \
  --is_training 1 \
  --root_path ./dataset/ClimSim \
  --model_id 1d \
  --model $model_name \
  --data ClimSim1D \
  --in_channel 25 \
  --out_channel 14 \
  --dropout 0.2 \
  --n_layers 4 \
  --d_model 1792 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10 \
  --batch_size 1600 \
  --patience 3 \
  --learning_rate 0.0001 \
  --lradj fix \
  --inverse \
  --bidirectional \
  --use_multi_gpu \
  --devices 0,1 \
  --postprocess \
  --sample_rate 0.31