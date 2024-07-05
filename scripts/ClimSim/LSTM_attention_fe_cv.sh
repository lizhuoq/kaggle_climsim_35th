#!/bin/bash

module load anaconda/2020.11
module load cuda/11.8
source activate climsim

model_name=LSTM_attention

for i in {1..5}
do
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
    --n_layers 3 \
    --d_model 768 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 30 \
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
    --optimizer adamw \
    --warmup_epochs 3 \
    --late_dropout_epoch 10 \
    --late_dropout_p 0.1 \
    --cv_fold $i
done
