#!/bin/bash

module load anaconda/2020.11
module load cuda/11.8
source activate climsim

model_name=Transformer

python -u run.py \
  --task_name fe \
  --is_training 1 \
  --root_path ./dataset/ClimSim \
  --model_id fe \
  --model $model_name \
  --data ClimSim1D \
  --in_channel 33 \
  --out_channel 14 \
  --dropout 0 \
  --n_layers 12 \
  --d_model 256 \
  --d_ff 512 \
  --nhead 8 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 50 \
  --batch_size 8000 \
  --patience 3 \
  --learning_rate 5e-4 \
  --lradj cosine \
  --inverse \
  --use_multi_gpu \
  --devices 0,1,2 \
  --postprocess \
  --add_feature_engineering 
