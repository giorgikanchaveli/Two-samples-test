# Two-samples-test

## Code structure

In the simulations, what we want to estimate is the **rejection rate**: run a two-sample test many times and see how often it rejects the null hypothesis. The main flow is the following:

- Generate hierarchical samples.
- Decide rejection.

**Generating hierarchical samples**

To define methods for generating hierarchical samples, we first need a struct for the laws of Random Probability Measures (RPMs). This is what `structures.jl` provides: `LawRPM`, the abstract type every law of RPM subtypes, so that generator functions can accept "any law of RPM" generically. Since other methods — like computing distance, p-values, etc. — also need to work on a hierarchical sample, `structures.jl` provides a struct for that too: `HierSample`, which holds a generated hierarchical sample (an n×m matrix of atoms, plus the interval [a,b] they live in).

With those types in place, `distributions.jl` defines the concrete laws of RPM (`tnormal_normal`, `discr_normal`, `mixture`, `DP`, etc.) as subtypes of `LawRPM`, and, for each one, a `sample_exch_seq` method that draws one exchangeable sequence from it — stacking n of these produces the `HierSample` the tests need.

**Deciding rejection**

Deciding whether to reject also requires computing a distance (test statistic) between two samples, so the `distances/` folder provides that: `distances/hipm.jl` computes HIPM (via `dlip`), and `distances/wow.jl` computes the Wasserstein / Wasserstein-over-Wasserstein distance used by the WoW test.

`methods.jl` then combines sample generation (`distributions.jl`) and distance computation (`distances/`) to implement the tests themselves — p-values, thresholds, and reject/don't-reject decisions for HIPM, WoW, DM (Fréchet) and Energy — and exposes the `rejection_rate_*` functions that repeat the whole simulate-then-test procedure S times and report how often each test rejects. This is exactly what the `simulations/` scripts call.

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

## Running mean/variance/counterexample simulations

We have three scripts that all share the same simulation logic: estimate how often the HIPM, WoW, DM and Energy two-sample tests reject the null hypothesis as the two samples' underlying laws of RPM, Q_1 and Q_2, are made to differ. They only differ in what Q_1/Q_2 are and what single parameter is swept over a grid:

- **`simulations/mean.jl`** — Q_1 = tnormal_normal(0.0, 0.5, -10.0, 10.0) is fixed, and Q_2 = tnormal_normal(δ, 0.5, -10.0, 10.0) for δ ranging over a grid in [-1.0, 1.0]. At δ = 0 the null hypothesis is true (Q_1 = Q_2), otherwise it's false.
- **`simulations/variance.jl`** — Q_1 = tnormal_normal(0.0, 0.2, -10.0, 10.0) is fixed, and Q_2 = tnormal_normal(0.0, 0.2*τ, -10.0, 10.0) for τ ranging over a grid in [0.1, 2.5]. At τ = 1 the null hypothesis is true (Q_1 = Q_2), otherwise it's false.
- **`simulations/counterexample.jl`** — Q_1 = discr_normal([-1.0, 1.0], [0.5, 0.5]) is fixed, and Q_2 = mixture(Q_1, Q_aux, λ) = λ*Q_1 + (1-λ)*Q_aux, where Q_aux = discr_normal([-2.0, 0.0, 2.0], [1/8, 3/4, 1/8]) and λ ranges over a grid in [0.0, 1.0]. At λ = 1 the null hypothesis is true (Q_2 = Q_1); at λ = 0, Q_2 = Q_aux and the two laws differ the most.

Each script has a detailed comment block at the top of the file (including all its command-line arguments) — this README just gives the quick-start version.

### How to run locally

**From a terminal**, pass arguments after the file name:

```bash
julia simulations/mean.jl --n 20 --m 30 --S 500 --n_samples 200
julia simulations/variance.jl --n 20 --m 30 --S 500 --n_samples 200
julia simulations/counterexample.jl --n 20 --m 30 --S 500 --n_samples 200
```

Any argument you leave out falls back to its default value. Run with `--help` to see all options (e.g. `julia simulations/mean.jl --help`).

**From VS Code / the Julia REPL** (e.g. `include("simulations/mean.jl")`), no arguments are passed, so every argument uses its default value. To change what runs, edit the `default = ...` values inside `parse_commandline()` in the corresponding file, then run it.

Results are saved as CSV files in `cluster_outputs/values/`, with columns `<δs|τs|λs>`, `hipm`, `wow`, `dm`, `energy`. The filename encodes the parameters used, e.g. `mean_n=20_m=30_S=500_bootstrap=false_n_samples=200.csv`.

### Running on the cluster

`shfiles_cluster/mean.sh`, `shfiles_cluster/variance.sh` and `shfiles_cluster/counterexample.sh` are SLURM batch scripts that run the corresponding simulation file on the cluster:

1. Edit the `configs` array at the top of the `.sh` file — each entry is one parameter combination (`"n m S n_samples"`), and each entry runs as a separate SLURM array task (`#SBATCH --array=1-N`, N = number of configs).
2. From the login node, make sure the project environment is instantiated/precompiled once (see the commented-out line near the top of the `.sh` file).
3. Submit the job with `sbatch shfiles_cluster/mean.sh` (or `variance.sh`/`counterexample.sh`). `JULIA_NUM_THREADS` is set automatically from `--cpus-per-task`, so the simulation runs multithreaded.
4. Each task writes its own CSV into `cluster_outputs/values/` on the cluster, exactly as it would locally.

### Plotting the results

**Important:** after simulation finishes, you need to manually move/copy the resulting CSV file from `cluster_outputs/values/` into `values/comparison_all/`. If the simulation was run on the cluster, this means transferring the file to your local computer first — e.g. using the FileZilla application.

`values_to_plots/plot_mean.jl`, `plot_variance.jl` and `plot_counterexample.jl` each read the matching CSV from `values/comparison_all/` and plot HIPM, WoW, DM and Energy rejection rate against δ/τ/λ, saving one PNG into `plots/comparison_all/`. Just run the file — see the comment block at the top of each for details.

## Rademacher threshold simulation

`simulations/section_rademacher.jl` checks how tight the theoretical Rademacher-complexity threshold for HIPM is. For fixed Q_1 = DP(1.0, Uniform(0.0, 1.0)) and Q_2 = DP(1.0, Beta(2.0, 2.0)) and θ = 0.05, it compares the empirical (1-θ)-quantile of dlip(Q_1, Q_2) (estimated by simulation) against the Rademacher threshold, across a grid of n = m, and also plots the theoretical threshold alone over a much larger grid to show its long-run decay.

### How to run locally

```bash
julia simulations/section_rademacher.jl --S 100 --n_rerun 5
```

Any argument you leave out falls back to its default value. Run with `--help` to see all options.

From VS Code / the Julia REPL (e.g. `include("simulations/section_rademacher.jl")`), no arguments are passed, so every argument uses its default value — edit the `default = ...` values inside `parse_commandline()` to change what runs.

Results are saved as two CSV files in `values/rademacher_section/`: `rademacher_emp_vs_theor_S=<S>.csv` (columns `ns`, `empirical`, `theoretical`) and `rademacher_thresholds.csv` (columns `ns`, `theoretical`). This script doesn't run on the cluster.

### Plotting the results

`values_to_plots/plot_rademacher.jl` reads both CSVs directly from `values/rademacher_section/` (no move/copy step needed) and saves the comparison plot, the solo decay plot, and both combined side by side to `plots/rademacher_section/`. Just run the file.

## Mortality dataset analysis

Besides the simulations, `mortality/` applies the tests to real data from the Human Mortality Database: for each year 1960–2010 and each gender, it tests whether Soviet Union influence countries and the rest have the same distribution of age-at-death.

- **`mortality/data_extractors.jl`** — loads the per-country life tables from `mortality/dataset/<gender>/` and turns them into the hierarchical samples / PMFs the other two scripts need. It's a library, not something you run directly.
- **`mortality/eda.jl`** — exploratory sanity-check plots (data lost under age truncation, individual country PMFs, PMF/quantile/pooled-PMF comparisons between the two groups).
- **`mortality/pvalues.jl`** — the actual test: computes p-values via HIPM, WoW, and two baselines (averaging, pooling) for each year/gender.



`pvalues.jl` saves one file per gender/method to `values/mortality_dataset/pvalues_<gender>_<method>.txt`. `values_to_plots/plot_pvalues.jl` reads directly from that folder (everything here runs locally, so there's no cluster move/copy step) and saves one PNG per gender to `plots/mortality_dataset/`.

## Cluster run times

Cluster times for current parameters:

counterexample: 171064 seconds
variance: 251215 seconds
mean: 292392 seconds
tp_4_4: 13888 seconds
fp_4: 14446 seconds
power_diff n = 100 m = 100: 226210 seconds