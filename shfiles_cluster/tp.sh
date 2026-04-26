#!/bin/bash -l
#SBATCH --job-name=tp                        # name of the SLURM job (shows up in queue)
#SBATCH --partition=compute                       # which partition/queue to use
#SBATCH --ntasks=1                                # number of tasks (1 is fine for Julia)
#SBATCH --cpus-per-task=20  					  # number of cores
#SBATCH --array=1-1                               # To run several jobs
#SBATCH --output=/home/3049277/logs/%x_%A_%a.out  # standard output log file
#SBATCH --error=/home/3049277/logs/%x_%A_%a.err   # standard error log file
#SBATCH --chdir=/home/3049277/Two-samples-test # working directory for the job

set -Eeuo pipefail                             # safer bash settings: stop on error, unset vars, etc.



echo "Running on node: $(hostname)"
echo "Running from: $(pwd)"                    # print the current working directory for confirmation



# --- Simulation parameter configuration ---
# Format: "label_q n m S n_perm"
configs=(
    "3 5 100 10 1000 100"
)

# Extract parameters for the current task (0-indexed array)
params=(${configs[$SLURM_ARRAY_TASK_ID-1]})
label_q_1=${params[0]}
label_q_2=${params[1]}
n=${params[2]}
m=${params[3]}
S=${params[4]}
n_perm=${params[5]}


export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK # set number of threads for julia

# Warning: write this in login node before running sh file.
# julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); println("Threads.nthreads() = ", Threads.nthreads())'
# IMPORTANT: Use the flags defined in your Julia parse_commandline() function
julia --project=. simulations/tp.jl --label_q_1 ${label_q_1} --label_q_2 ${label_q_2} --n ${n} --m ${m} --S ${S} --n_perm ${n_perm}

echo "This is the end"                         # simple marker showing the script reached the end successfully
