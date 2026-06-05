#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --output=/cluster/scratch/gsialelli/logs/compute_AEF_stats-%j.txt
#SBATCH --error=/cluster/scratch/gsialelli/logs/compute_AEF_stats-%j.txt
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=stats

python compute_statistics.py