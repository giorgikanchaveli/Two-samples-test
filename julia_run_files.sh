#!/bin/bash -l
#SBATCH --job-name=run_julia_file              # name of the SLURM job (shows up in queue)
#SBATCH --partition=compute                    # which partition/queue to use
#SBATCH --ntasks=1                           # number of tasks (1 is fine for Julia)
#SBATCH --cpus-per-task=32						# number of cores
#SBATCH --time=00:05:00                        # maximum run time (hh:mm:ss)
#SBATCH --output=/home/3049277/logs/%x_%j.out  # standard output log file
#SBATCH --error=/home/3049277/logs/%x_%j.err   # standard error log file
#SBATCH --chdir=/home/3049277/Two-samples-test # working directory for the job

set -Eeuo pipefail                             # safer bash settings: stop on error, unset vars, etc.

# Activate conda env that contains the R you built RCall against
# (This ensures the R and its libraries from the conda environment are available)
module load miniconda3                         # load the conda module on the cluster
eval "$(conda shell.bash hook)"                # enable conda commands inside batch scripts
conda activate Renv2                           # activate the conda environment (named Renv2)

echo "Running from: $(pwd)"                    # print the current working directory for confirmation

# Run the Julia script using the environment defined by Project.toml in this directory
# (gio.jl can contain 'using RCall' or any other Julia code)
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); println("Threads.nthreads() = ", Threads.nthreads())'
#julia --project=. n_vs_m_hipm.jl

echo "This is the end"                         # simple marker showing the script reached the end successfully
