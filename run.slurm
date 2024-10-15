#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --job-name="pytorch-lightning-demo"
#SBATCH --output=pytorch-lightning-demo.out


cd ${SLURM_SUBMIT_DIR}/

module purge
module load Anaconda3/2023.09-0
conda activate pl_env

srun python trainer.py