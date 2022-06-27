#!/bin/bash

epochs=1
batch_size=128
fusion_size=$1

src_dataset='de_core_news_sm'
tgt_dataset='en_core_web_sm'


CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --epochs $epochs \
  --batch-size $batch_size \
  --source-dataset $src_dataset \
  --target-dataset $tgt_dataset \
  --fusion-size $fusion_size 
