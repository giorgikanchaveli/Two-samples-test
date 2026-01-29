#!/bin/bash -l
#SBATCH --job-name=variance                        # name of the SLURM job (shows up in queue)
#SBATCH --partition=compute                       # which partition/queue to use
#SBATCH --ntasks=1                                # number of tasks (1 is fine for Julia)
#SBATCH --cpus-per-task=20  					  # number of cores
#SBATCH --time=13:00:00                           # maximum run time (hh:mm:ss)
#SBATCH --array=1-1                               # To run several jobs
#SBATCH --output=/home/3049277/logs/%x_%A_%a.out  # standard output log file
#SBATCH --error=/home/3049277/logs/%x_%A_%a.err   # standard error log file
#SBATCH --chdir=/home/3049277/Two-samples-test # working directory for the job

set -Eeuo pipefail                             # safer bash settings: stop on error, unset vars, etc.

# Activate conda env that contains the R you built RCall against
# (This ensures the R and its libraries from the conda environment are available)
module load miniconda3                         # load the conda module on the cluster
eval "$(conda shell.bash hook)"                # enable conda commands inside batch scripts
conda activate Renv2                           # activate the conda environment (named Renv2)


echo "Running on node: $(hostname)"
echo "Running from: $(pwd)"                    # print the current working directory for confirmation



# --- Simulation parameter configuration ---
# Format: "n m S n_samples"
configs=(
    "35 100 90 45"
)

# Extract parameters for the current task (0-indexed array)
params=(${configs[$SLURM_ARRAY_TASK_ID-1]})

n=${params[0]}
m=${params[1]}
S=${params[2]}
n_samples=${params[3]}


export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK # set number of threads for julia

# Warning: write this in login node before running sh file.
# julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); println("Threads.nthreads() = ", Threads.nthreads())'
# IMPORTANT: Use the flags defined in your Julia parse_commandline() function
julia --project=. simulations/variance.jl --n ${n} --m ${m} --S ${S} --n_samples ${n_samples}

echo "This is the end"                         # simple marker showing the script reached the end successfully
