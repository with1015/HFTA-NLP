#!/bin/bash

batch_size=$1

if [ $# -ne 1 ]; then
  echo "[USAGE] [batch size]"
  exit 1
fi
echo "batch size: "$batch_size

master=shark8
echo 'Storage:' $storage

data_dir=/ssd_dataset/dataset/pytorch_wmt16_en_de/

visible_gpus='0'
local_batch=$batch_size
epochs=1
num_layers=4
echo "Number of layers: "$num_layers

fusion_size=2

CUDA_VISIBLE_DEVICES=$visible_gpus python train.py \
  --dist-url 'tcp://127.0.0.1:20000' \
  --dist-backend 'nccl' \
  --world-size 1 \
  --rank 0 \
  --gpu 0 \
  --dataset-dir $data_dir \
  --math fp32 \
  --seed 2 \
  --num-layers $num_layers \
  --train-batch-size $local_batch \
  --epochs $epochs \
  --fusion-size $fusion_size
