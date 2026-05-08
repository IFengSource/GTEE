#!/usr/bin/env bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/imputation" ]; then
    mkdir ./logs/imputation
fi

# ETTh1
model_name=SAITS_GTEE
dataset_name=ETT_h1
dataset_fold=data/generated_datasets/ETTh1/ett_rate05_step48_point
rate=05
miss_type=point
save_path=results/$miss_type'_rate_'$rate
device_name=cuda:0

echo "experiment time: $(date +%Y-%m-%d\ %H:%M:%S)" > logs/imputation/$model_name'_'$dataset_name'_'$miss_type'_'$rate.log
python train_model_GTEE.py \
    --model $model_name \
    --dataset $dataset_name \
    --dataset_fold_path $dataset_fold \
    --saving_path $save_path \
    --device $device_name \
    >> logs/imputation/$model_name'_'$dataset_name'_'$miss_type'_'$rate.log 2>&1

model_name=SAITS_GTEE
dataset_name=ETT_h1
dataset_fold=data/generated_datasets/ETTh1/ett_rate03_step48_block_blocklen6
rate=05
miss_type=block
save_path=results/$miss_type'_rate_'$rate
device_name=cuda:0

echo "experiment time: $(date +%Y-%m-%d\ %H:%M:%S)" > logs/imputation/$model_name'_'$dataset_name'_'$miss_type'_'$rate.log
python train_model_GTEE.py \
    --model $model_name \
    --dataset $dataset_name \
    --dataset_fold_path $dataset_fold \
    --saving_path $save_path \
    --device $device_name \
    >> logs/imputation/$model_name'_'$dataset_name'_'$miss_type'_'$rate.log 2>&1

model_name=SAITS_GTEE
dataset_name=ETT_h1
dataset_fold=data/generated_datasets/ETTh1/ett_rate05_step48_subseq_seqlen36
rate=05
miss_type=subseq
save_path=results/$miss_type'_rate_'$rate
device_name=cuda:0

echo "experiment time: $(date +%Y-%m-%d\ %H:%M:%S)" > logs/imputation/$model_name'_'$dataset_name'_'$miss_type'_'$rate.log
python train_model_GTEE.py \
    --model $model_name \
    --dataset $dataset_name \
    --dataset_fold_path $dataset_fold \
    --saving_path $save_path \
    --device $device_name \
    >> logs/imputation/$model_name'_'$dataset_name'_'$miss_type'_'$rate.log 2>&1
