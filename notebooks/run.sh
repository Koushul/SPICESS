#!/bin/bash
#SBATCH --job-name=wandb
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --ntasks=5
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --cluster=gpu
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00

export OMP_NUM_THREADS=5

srun wandb agent anticancer/spicess/5rkyy0my
srun wandb agent anticancer/spicess/5rkyy0my
srun wandb agent anticancer/spicess/5rkyy0my
srun wandb agent anticancer/spicess/5rkyy0my
srun wandb agent anticancer/spicess/5rkyy0my
