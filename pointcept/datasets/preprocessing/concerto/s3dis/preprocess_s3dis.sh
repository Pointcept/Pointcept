#!/bin/bash

dataset_root=""
output_root=""
num_workers=6  
parse_depths=false
parse_pointclouds=false

while getopts "d:o:n:r:pc" opt; do
  case $opt in
    d) dataset_root=$OPTARG ;;  
    o) output_root=$OPTARG ;;   
    n) num_workers=$OPTARG ;;   
    r) raw_root=$OPTARG ;;  
    p) parse_depths=true ;; 
    c) parse_pointclouds=true ;; 
    *) echo "Usage: $0 -d <dataset_root> -o <output_root> -r <raw_root> [-n <num_workers>] [-p] [-c]"; exit 1 ;;
  esac
done

if [ -z "$dataset_root" ] || [ -z "$output_root" ] || [ -z "$raw_root" ]; then
    echo "Usage: $0 -d <dataset_root> -o <output_root> -r <raw_root> [-n <num_workers>] [-p] [-c]"
    exit 1
fi

for i in $(seq 1 $((num_workers))); do
    cmd="python pointcept/datasets/preprocessing/concerto/s3dis/preprocess_s3dis.py --splits Area_$i \
    --num_workers 1 \
    --thread_id 0 \
    --dataset_root $dataset_root \
    --output_root $output_root \
    --raw_root $raw_root \
    --parse_normal"
    if $parse_depths; then
        cmd="$cmd --parse_depths"
    fi

    if $parse_pointclouds; then
        cmd="$cmd --parse_pointclouds"
    fi

    eval "$cmd &"
done

wait
