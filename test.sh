#!/bin/bash -l
#SBATCH --job-name=gpu_test                # job name
#SBATCH --partition=slurm-general          # specifying which partition to run job on, if omitted default partition will be used (slurm-general)
#SBATCH --account=slurmgeneral             # only applicable if user is assigned multiple accounts
#SBATCH --ntasks=1                         # commands to run in parallel
#SBATCH --time=1:00:00                     # time limit on the job
#SBATCH --mem=1gb                          # request 1gb of memory
#SBATCH --output=gpu_test.log              # output and error log

date
python3 test.py