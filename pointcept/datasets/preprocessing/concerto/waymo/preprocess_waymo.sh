#!/bin/bash

dataset_root=""
output_root=""
num_workers=16
splits="training validation"

while getopts "d:o:n:s:" opt; do
  case $opt in
    d) dataset_root=$OPTARG ;;
    o) output_root=$OPTARG ;;
    n) num_workers=$OPTARG ;;
    s) splits=$OPTARG ;;
    *) echo "Usage: $0 -d <dataset_root> -o <output_root> -s '<splits>' [-n <num_workers>]"; exit 1 ;;
  esac
done

if [ -z "$dataset_root" ] || [ -z "$output_root" ]; then
    echo "Usage: $0 -d <dataset_root> -o <output_root> -s '<splits>' [-n <num_workers>]"
    exit 1
fi

for i in $(seq 0 $((num_workers - 1))); do
    cmd="python pointcept/datasets/preprocessing/concerto/waymo/preprocess_waymo.py \
        --dataset_root $dataset_root \
        --output_root $output_root \
        --splits $splits \
        --num_workers $num_workers \
        --thread_id $i"
    
    eval "$cmd" > "preprocess_waymo_thread_${i}.log" 2>&1 &
done

wait
