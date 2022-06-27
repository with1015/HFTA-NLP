#!/bin/bash

loop=$1

for((i=0; i < ${loop}; i++))
do
  ./single_gpu_run.sh 32 &
done
