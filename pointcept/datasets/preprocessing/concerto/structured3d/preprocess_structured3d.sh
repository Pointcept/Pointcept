#!/bin/bash

dataset_root=""
output_root=""
num_workers=16  
parse_depths=false
parse_pointclouds=false

while getopts "d:o:n:pc" opt; do
  case $opt in
    d) dataset_root=$OPTARG ;;  
    o) output_root=$OPTARG ;;   
    n) num_workers=$OPTARG ;;   
    p) parse_depths=true ;;
    c) parse_pointclouds=true ;;
    *) echo "Usage: $0 -d <dataset_root> -o <output_root> [-n <num_workers>] [-p] [-c]"; exit 1 ;;
  esac
done

if [ -z "$dataset_root" ] || [ -z "$output_root" ]; then
    echo "Usage: $0 -d <dataset_root> -o <output_root> [-n <num_workers>] [-p] [-c]"
    exit 1
fi

for i in $(seq 0 $((num_workers - 1))); do
    cmd="python pointcept/datasets/preprocessing/concerto/structured3d/preprocess_structured3d.py --thread_id $i \
    --num_workers $num_workers \
    --dataset_root $dataset_root \
    --output_root $output_root \
    --fuse_prsp --fuse_pano"
    
    if $parse_depths; then
        cmd="$cmd --parse_depths"
    fi

    if $parse_pointclouds; then
        cmd="$cmd --parse_pointclouds"
    fi

    eval "$cmd &"
done

wait
