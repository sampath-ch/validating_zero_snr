#!/bin/bash

#SBATCH --job-name=sd_lora_train
#SBATCH --output=/scratch/schettip/CS757/%u/%x-%N-%j.out
#SBATCH --error=/scratch/schettip/CS757/%u/%x-%N-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-12:00:00        ## Changed to 12 hours
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=6       ## Matched to DataLoader num_workers
#SBATCH --mem=64G               ## Explicitly defined Gigabytes
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mail-user=schettip@gmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

## 1. Load Modules

module load cuda/12.6

cd /scratch/schettip/CS757

source lora_env/bin/activate

module load python/3.10

module load cuda/12.6
module load cudnn



## 3. Silent W&B Authentication (No interactive login required)
export WANDB_API_KEY="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"

## 4. Run the Script
python improved_train.py