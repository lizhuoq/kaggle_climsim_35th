#!/bin/bash

module load anaconda/2020.11
module load cuda/11.8
source activate climsim

model_name=Transformer

python -u run.py \
  --task_name 1d \
  --is_training 1 \
  --root_path ./dataset/ClimSim \
  --feature_scaler_path ./dataset/ClimSim/feature_scaler.joblib \
  --target_scaler_path ./dataset/ClimSim/target_scaler.joblib \
  --weight_path ./dataset/ClimSim/sample_submission.parquet \
  --model_id 1d \
  --model $model_name \
  --data ClimSim1D \
  --levels 60 \
  --vertical_in_channel 9 \
  --vertical_out_channel 6 \
  --scalar_in_channel 16 \
  --scalar_out_channel 8 \
  --dropout 0.2 \
  --n_layers 6 \
  --d_model 256 \
  --d_ff 256 \
  --nhead 4 \
  --activation relu \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10 \
  --batch_size 16000 \
  --patience 3 \
  --learning_rate 0.0005 \
  --use_multi_gpu \
  --devices 0,1 \
  --lradj fix \
  --inverse