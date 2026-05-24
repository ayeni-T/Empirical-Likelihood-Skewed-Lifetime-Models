#!/bin/bash
#SBATCH --job-name=QTEG_JEL
#SBATCH --account=math161r53
#SBATCH --partition=qCPU120
#SBATCH --array=0-35
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=120:00:00
#SBATCH --output=/home/users/tayeni2/logs/qteg_%A_%a.out
#SBATCH --error=/home/users/tayeni2/logs/qteg_%A_%a.err
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=tayeni2@gsu.edu

mkdir -p /home/users/tayeni2/logs
mkdir -p /home/users/tayeni2/results

module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate /home/users/tayeni2/qteg_env

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

python /home/users/tayeni2/QTEG_JEL_Arctic.py --block ${SLURM_ARRAY_TASK_ID}

echo "Block ${SLURM_ARRAY_TASK_ID} completed at $(date)"
