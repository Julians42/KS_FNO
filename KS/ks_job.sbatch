#!/bin/bash

#SBATCH --job-name=train_ks
#SBATCH --output=train_ks.out
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --error=train_ks.out


python train_ks.py
