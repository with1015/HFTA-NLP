#!/bin/bash

batch_size=128
fusion_size=3

src_dataset='de_core_news_sm'
tgt_dataset='en_core_web_sm'

python3 profile_main.py \
  --batch-size $batch_size \
  --source-dataset $src_dataset \
  --target-dataset $tgt_dataset \
  --fusion-size $fusion_size
