#!/bin/bash
#SBATCH --job-name=pymc5_job
#SBATCH --output=output_%j.log
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=recovery_scores_%A.out
#SBATCH --partition=owner_tam
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haagth@staff.uni-marburg.de


# purge and load environment
module purge
module load miniconda
source $CONDA_ROOT/bin/activate

conda activate pymc5_env

# run the script
python3 -O recovery_per_timesteps_block_data_timesteps.py --num_procs=$SLURM_CPUS_PER_TASK
