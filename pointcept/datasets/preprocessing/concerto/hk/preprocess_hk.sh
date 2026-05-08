#!/bin/bash

dataset_root=""
output_root=""
num_workers=16  
parse_depths=false

while getopts "d:o:n:p" opt; do
  case $opt in
    d) dataset_root=$OPTARG ;;  
    o) output_root=$OPTARG ;;   
    n) num_workers=$OPTARG ;;   
    *) echo "Usage: $0 -d <dataset_root> -o <output_root> [-n <num_workers>]"; exit 1 ;;
  esac
done

if [ -z "$dataset_root" ] || [ -z "$output_root" ]; then
    echo "Usage: $0 -d <dataset_root> -o <output_root> [-n <num_workers>]"
    exit 1
fi

for i in $(seq 0 $((num_workers - 1))); do
    cmd="python pointcept/datasets/preprocessing/concerto/hk/preprocess_hk.py --thread_id $i \
    --num_workers $num_workers \
    --dataset_root $dataset_root \
    --output_root $output_root"
    eval "$cmd &"
done

wait
