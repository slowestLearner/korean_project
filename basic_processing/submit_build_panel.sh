#!/bin/bash
#SBATCH --job-name=build_panel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=notchpeak
#SBATCH --output=build_panel_%j.log

# Load miniconda and activate your environment
module load miniconda3
conda activate base 
source /uufs/chpc.utah.edu/sys/installdir/r8/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate korean_env

# Navigate to code directory
cd /uufs/chpc.utah.edu/common/home/u6074436/korean_project/basic_processing

# Run the python script
python -u 4_build_panel.py