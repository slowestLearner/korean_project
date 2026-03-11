#!/bin/bash
#SBATCH --job-name=lasso_theme_m
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=notchpeak
#SBATCH --output=lasso_theme_monthly_%j.log

module load miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate korean_env

cd /uufs/chpc.utah.edu/common/home/u6074436/korean_project/basic_processing

python -u 3_lasso_theme_monthly.py