#!/bin/bash

module load anaconda/2020.11
module load cuda/11.8
source activate climsim

model_name=LSTM

python -u run.py \
  --task_name fe_v1 \
  --is_training -2 \
  --root_path ./dataset/ClimSim \
  --model_id fe_v1 \
  --model $model_name \
  --data ClimSim1D \
  --in_channel 39 \
  --out_channel 14 \
  --dropout 0.2 \
  --n_layers 4 \
  --d_model 768 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10 \
  --batch_size 1600 \
  --patience 3 \
  --learning_rate 1e-4 \
  --lradj cosine \
  --inverse \
  --bidirectional \
  --use_multi_gpu \
  --devices 0,1,2 \
  --postprocess \
  --add_feature_engineering \
  --add_fe_v1