#!/bin/bash

cam_root=""
point_cloud_root=""
output_root=""
num_workers=16

while getopts "c:p:o:n" opt; do
  case $opt in
    c) cam_root=$OPTARG ;;
    p) point_cloud_root=$OPTARG ;;
    o) output_root=$OPTARG ;;
    n) num_workers=$OPTARG ;;
    *) echo "Usage: $0 -c <cam_root> -p <point_cloud_root> -o <output_root> [-n <num_workers>]"; exit 1 ;;
  esac
done

if [ -z "$cam_root" ] || [ -z "$point_cloud_root" ] || [ -z "$output_root" ]; then
    echo "Error: Missing required arguments. Use -h for help."
    echo "Usage: $0 -c <cam_root> -p <point_cloud_root> -o <output_root> [-n <num_workers>]"
    exit 1
fi

for i in $(seq 0 $((num_workers - 1))); do
    cmd="python pointcept/datasets/preprocessing/concerto/cap3d/preprocess_cap3d.py \
    --cam_root $cam_root \
    --point_cloud_root $point_cloud_root \
    --output_root $output_root \
    --num_workers $num_workers \
    --thread_id $i"

    eval "$cmd &"
done

wait
