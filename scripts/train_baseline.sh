#!/bin/bash
#SBATCH --account=nexus
#SBATCH --job-name=baseline
#SBATCH --time=3-0:00:00
#SBATCH --partition=tron
#SBATCH --qos=default
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-%j-%x.out
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail         
#SBATCH --mail-user=psando@umd.edu

#--SBATCH --array=0-1
#--SBATCH --dependency=afterok:
#--SBATCH --mail-type=end          
#--SBATCH --mail-type=fail         
#--SBATCH --mail-user=psando@umd.edu

#--SBATCH --output /dev/null
#--SBATCH --output=slurm-%j-%x.out

run_name="baseline"

# 10 steps takes ~1 minute, so 600 steps in 1 hour, 14400 steps in 24 hours, 28800 steps in 48 hours
python train.py config/train_gpt2.py \
                --wandb_project="owt-2025-30" \
                --wandb_run_name=${run_name} \
                --out_dir="/fs/nexus-scratch/psando/nanoGPT-experiments/out-${run_name}-${SLURM_JOB_ID}" \
                --data_dir="/fs/nexus-scratch/psando/owt" \
                --log_interval=100 \
                --save_checkpoint_every=14000