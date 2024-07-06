#!/bin/bash

module load anaconda/2020.11
module load cuda/11.8
source activate climsim

model_name=CNN_Transformer
i=1

python -u run.py \
    --task_name fecv \
    --is_training 1 \
    --root_path ./dataset/ClimSim \
    --model_id fecv \
    --model $model_name \
    --data ClimSim1D \
    --in_channel 33 \
    --out_channel 14 \
    --dropout 0 \
    --n_layers 4 \
    --d_model 384 \
    --d_ff 1536 \
    --nhead 8 \
    --drop_path 0.2 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 50 \
    --batch_size 1600 \
    --patience 3 \
    --learning_rate 1e-4 \
    --lradj cosine \
    --inverse \
    --use_multi_gpu \
    --devices 0,1,2 \
    --postprocess \
    --add_feature_engineering \
    --optimizer adamw \
    --warmup_epochs 3 \
    --late_dropout_epoch 20 \
    --late_dropout_p 0.1 \
    --cv_fold $i

