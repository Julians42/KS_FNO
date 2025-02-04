#!/bin/bash

#SBATCH --job-name=train_ks
#SBATCH --output=train_ks_%A.out
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

echo $config_file

python ../train_ks.py --config_file "$config_file"
