#!/bin/bash

#SBATCH --job-name=train_ks
#SBATCH --output=train_ks.out
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

python train_ks.py
