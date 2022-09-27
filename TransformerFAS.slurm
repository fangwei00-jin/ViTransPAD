#!/bin/bash
#SBATCH -J TransFAS
#SBATCH --partition=guests
#SBATCH --ntasks-per-node=1              ## number of tasks 
#SBATCH --cpus-per-task=4       ## number of cores per task
#SBATCH --mem-per-cpu=30G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A40:1


echo "Just checking!
Starting job on execution node : $(hostname)
Date: $(date)
Waiting ..."
echo "... Running"

# load the modules that you need: check them with `module avail`
# module load nvidia/cuda/11.0
# module load python/3.7

# load env
# module load conda/anaconda3
#conda init bash
#source activate

# activate conda env: py37 is one of my envs
# source activate py3_torch1.8

# run script
python ~/code/Transfomer_FAS/train.py --config configs/OuluNPU.json -i tcp://localhost:12346

