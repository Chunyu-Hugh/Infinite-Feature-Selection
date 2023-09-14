#!/bin/bash -l
#SBATCH --job-name=frommaolei                # job name
#SBATCH --account=slurmgeneral             # only applicable if user is assigned multiple accounts
#SBATCH --ntasks=1                         # commands to run in parallel
#SBATCH --output=frommaolei.log              # output and error log
#SBATCH --error=frommaolei.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb                          # request 1gb of memory
#SBATCH --nodelist=slurm-a6000-wang
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chu034@odu.edu
./run.sh
