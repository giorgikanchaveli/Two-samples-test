#!/bin/bash -l
#SBATCH --job-name=power_diff                     # name of the SLURM job (shows up in queue)
#SBATCH --partition=defq                          # which partition/queue to use
#SBATCH --ntasks=1                                # number of tasks (1 is fine for Julia)
#SBATCH --cpus-per-task=20                        # number of cores
#SBATCH --array=1-4                               # To run several jobs
#SBATCH --output=/home/3049277/logs/%x_%A_%a.out  # standard output log file
#SBATCH --error=/home/3049277/logs/%x_%A_%a.err   # standard error log file
#SBATCH --chdir=/home/3049277/Two-samples-test    # working directory for the job

set -Eeuo pipefail                                # safer bash settings: stop on error, unset vars, etc.

echo "Running on node: $(hostname)"
echo "Running from: $(pwd)"

# --- Simulation parameter configuration ---
# Format: "n m S n_samples"
configs=(
    "20 100 50 70"
    "30 100 50 70"
    "50 100 50 70"
    "100 100 50 70"
)

# Extract parameters for the current task (0-indexed array)
params=(${configs[$SLURM_ARRAY_TASK_ID-1]})

n=${params[0]}
m=${params[1]}
S=${params[2]}
n_samples=${params[3]}

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK    # set number of threads for Julia

# Warning: write this in login node before running sh file.
# julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); println("Threads.nthreads() = ", Threads.nthreads())'
# IMPORTANT: Use the flags defined in your Julia parse_commandline() function
julia --project=. simulations/power_diff_save.jl --n ${n} --m ${m} --S ${S} --n_samples ${n_samples}

echo "This is the end"