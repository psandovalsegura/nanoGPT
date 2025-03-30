#!/bin/bash
#SBATCH --account=vulcan-djacobs
#SBATCH --job-name=prepare-owt-dataset
#SBATCH --time=5:00:00
#SBATCH --partition=vulcan-cpu
#SBATCH --qos=vulcan-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=%j-%x.out
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

# Takes 7 minutes if files already downloaded and only need to tokenize
python data/openwebtext/prepare.py
