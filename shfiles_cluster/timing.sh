#!/bin/bash -l
#SBATCH --job-name=timing                             # name of the SLURM job (shows up in queue)
#SBATCH --partition=defq                              # which partition/queue to use
#SBATCH --ntasks=1                                    # number of tasks
#SBATCH --cpus-per-task=1                             # single core: dlip and ww are both single-threaded
#SBATCH --output=/home/3049277/logs/%x_%j.out         # standard output log file
#SBATCH --error=/home/3049277/logs/%x_%j.err          # standard error log file
#SBATCH --chdir=/home/3049277/Two-samples-test        # working directory for the job

set -Eeuo pipefail                                    # safer bash settings: stop on error, unset vars, etc.

echo "Running on node: $(hostname)"
echo "Running from: $(pwd)"

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Warning: write this in login node before running sh file.
# julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); println("Threads.nthreads() = ", Threads.nthreads())'
julia --project=. distances/timing.jl

echo "This is the end"
