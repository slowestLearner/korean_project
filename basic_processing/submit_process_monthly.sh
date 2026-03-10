#!/bin/bash
#SBATCH --job-name=inst_monthly
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=notchpeak
#SBATCH --output=process_monthly.log


# Load miniconda and activate your environment
module load miniconda3
conda activate base 
source /uufs/chpc.utah.edu/sys/installdir/r8/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate korean_env


python 2_process_inst_monthly.py