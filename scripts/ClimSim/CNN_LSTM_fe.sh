#!/bin/bash

module load anaconda/2020.11
module load cuda/11.8
source activate climsim

model_name=CNN_LSTM

python -u run.py \
  --task_name fe \
  --is_training -2 \
  --root_path ./dataset/ClimSim \
  --model_id fe \
  --model $model_name \
  --data ClimSim1D \
  --in_channel 33 \
  --out_channel 14 \
  --dropout 0.2 \
  --n_layers 4 \
  --conv_out_channel 128 \
  --d_model 768 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10 \
  --batch_size 1600 \
  --patience 3 \
  --learning_rate 0.0001 \
  --lradj cosine \
  --inverse \
  --bidirectional \
  --use_multi_gpu \
  --devices 0,1,2 \
  --postprocess \
  --add_feature_engineering