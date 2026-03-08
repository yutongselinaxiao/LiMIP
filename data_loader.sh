#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --output=../slurm/%A-%a.out

module purge


/home1/xiaoyuto/.conda/envs/scip-clean/bin/python Cont_generate_dataset.py setcover_densize --density 0.2 -j 15

