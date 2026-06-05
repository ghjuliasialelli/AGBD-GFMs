#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --output=/cluster/scratch/gsialelli/logs/compute_TESSERA_stats_lite-%j.txt
#SBATCH --error=/cluster/scratch/gsialelli/logs/compute_TESSERA_stats_lite-%j.txt
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=stats_lite

python compute_statistics.py --lite --year 2020 --regions global
