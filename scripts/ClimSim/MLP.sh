#!/bin/bash

module load anaconda/2020.11
module load cuda/11.8
source activate climsim

model_name=MLP

python -u run.py \
  --task_name small \
  --is_training 1 \
  --root_path ./dataset/ClimSimSmall \
  --feature_scaler_path ./dataset/ClimSimSmall/features.joblib \
  --target_scaler_path ./dataset/ClimSimSmall/targets.joblib \
  --weight_path ./dataset/ClimSim/sample_submission.parquet \
  --model_id small \
  --model $model_name \
  --data ClimSimSmall \
  --in_channel 462 \
  --out_channel 305 \
  --dropout 0.3 \
  --n_layers 6 \
  --d_model 1024 \
  --d_ff 4096 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 30 \
  --batch_size 20000 \
  --patience 3 \
  --learning_rate 0.0006 \
  --lradj cosine \
  --inverse \
  --test_path ./dataset/ClimSim/test.parquet \
  --postprocess