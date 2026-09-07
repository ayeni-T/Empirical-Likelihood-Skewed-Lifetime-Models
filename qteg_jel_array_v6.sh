#!/bin/bash
#SBATCH --job-name=QTEG_JEL_v6
#SBATCH --account=<your_hpc_account>
#SBATCH --partition=<your_hpc_partition>
#SBATCH --array=0-35
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=120:00:00
#SBATCH --output=<path_to_repo>/logs/qteg_%A_%a.out
#SBATCH --error=<path_to_repo>/logs/qteg_%A_%a.err
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=<your_email>

mkdir -p <path_to_repo>/logs
mkdir -p <path_to_repo>/results

module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate <path_to_conda_env>

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "============================================"
echo "Block:   ${SLURM_ARRAY_TASK_ID}"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Python:  $(which python)"
echo "Started: $(date)"
echo "CPUs:    ${SLURM_CPUS_PER_TASK}"
echo "============================================"

python <path_to_repo>/QTEG_JEL_Arctic_v6.py --block ${SLURM_ARRAY_TASK_ID}

echo "Block ${SLURM_ARRAY_TASK_ID} completed at $(date)"
