#!/bin/bash

module load anaconda/2020.11
module load cuda/11.8
source activate climsim

model_name=Patch_LSTM

python -u run.py \
  --task_name fe \
  --is_training 1 \
  --root_path ./dataset/ClimSim \
  --model_id fe \
  --model $model_name \
  --data ClimSim1D \
  --in_channel 33 \
  --out_channel 14 \
  --patch_size 10 \
  --stride 5 \
  --num_patch 11 \
  --dropout 0.2 \
  --n_layers 3 \
  --d_model 64 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 50 \
  --batch_size 16000 \
  --patience 3 \
  --learning_rate 1e-3 \
  --lradj cosine \
  --inverse \
  --bidirectional \
  --use_multi_gpu \
  --devices 0,1,2 \
  --postprocess \
  --add_feature_engineering
