#!/bin/bash
#SBATCH --job-name=pymc5_job
#SBATCH --output=output_%j.log
#SBATCH --time=38:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --output=simulation_scores_%A.out
#SBATCH --partition=owner_tam
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haagth@staff.uni-marburg.de


# purge and load environment
module purge
module load miniconda
source $CONDA_ROOT/bin/activate

conda activate pymc5_env

# run the script
python3 -O master_loop_simulation.py --num_procs=$SLURM_CPUS_PER_TASK
