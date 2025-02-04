#!/bin/bash


for (( i=1; i<2; i++ )); do
    power=$(( 2**i ))

    config_file="/home/jschmitt/KS_FNO/KS/resolution_configs/resolution_$power.yaml"
    echo $config_file
    sbatch --export=ALL,config_file="$config_file" dispatcher.sh
done
