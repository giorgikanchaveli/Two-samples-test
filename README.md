# Two-samples-test

## Running FP/TP simulations

We have two scripts for estimating how often the HIPM and WoW two-sample tests reject the null hypothesis:

- **`simulations/fp.jl`** — False Positive Rate. Both samples are drawn from the *same* law of RPM, so the null hypothesis is true, and any rejection is a false positive.
- **`simulations/tp.jl`** — True Positive Rate (power). The two samples are drawn from *different* laws of RPM, so the null hypothesis is false, and a rejection is a true positive.

Both scripts share the same logic: simulate hierarchical samples many times, run the tests at a grid of significance levels θ = 0, 0.01, ..., 1, and save the estimated rejection rates to a CSV file. Each script has a detailed comment block at the top of the file (including all its command-line arguments) — this README just gives the quick-start version.

### How to run locally

**From a terminal**, pass arguments after the file name:

```bash
julia simulations/fp.jl --label_q 3 --n 20 --m 30 --S 500 --n_perm 200
julia simulations/tp.jl --label_q_1 1 --label_q_2 1 --n 20 --m 30 --S 500 --n_perm 200
```

Any argument you leave out falls back to its default value. Run with `--help` to see all options (e.g. `julia simulations/fp.jl --help`).

**From VS Code / the Julia REPL** (e.g. `include("simulations/fp.jl")`), no arguments are passed, so every argument uses its default value. To change what runs, edit the `default = ...` values inside `parse_commandline()` in the corresponding file, then run it.

Results are saved as CSV files in `cluster_outputs/values/`, with columns `θs`, `<prefix>_hipm`, `<prefix>_wow` (prefix `fp` or `tp`). The filename encodes the parameters used, e.g. `fp_label=3_n=20_m=30_S=500_n_perm=200.csv`.

### Running on the cluster

`shfiles_cluster/fp.sh` and `shfiles_cluster/tp.sh` are SLURM batch scripts that run `fp.jl`/`tp.jl` on the cluster:

1. Edit the `configs` array at the top of the `.sh` file — each entry is one parameter combination (e.g. for `fp.sh`: `"label_q n m S n_perm"`), and each entry runs as a separate SLURM array task (`#SBATCH --array=1-N`, N = number of configs).
2. From the login node, make sure the project environment is instantiated/precompiled once (see the commented-out line near the top of the `.sh` file).
3. Submit the job with `sbatch shfiles_cluster/fp.sh` (or `tp.sh`). `JULIA_NUM_THREADS` is set automatically from `--cpus-per-task`, so the simulation runs multithreaded.
4. Each task writes its own CSV into `cluster_outputs/values/` on the cluster, exactly as it would locally.


### Plotting the results


**Important:** after simulation finishes, you need to manually move/copy the resulting CSV files from `cluster_outputs/values/` into `values/fp_tp/`. If the simulation was run on the cluster, this means transferring the files to your local computer first — e.g. using the FileZilla application.

`values_to_plots/plot_tpfp_csv.jl` reads CSV files from `values/fp_tp/` and plots HIPM vs. WoW rejection rate against θ, saving one PNG per CSV into `plots/fp_tp/`. Just run the file — see the comment block at the top of `plot_tpfp_csv.jl` for details.

## Cluster run times

Cluster times for current parameters:

counterexample: 171064 seconds
variance: 251215 seconds
mean: 292392 seconds
tp_4_4: 13888 seconds
fp_4: 14446 seconds
power_diff n = 100 m = 100: 226210 seconds