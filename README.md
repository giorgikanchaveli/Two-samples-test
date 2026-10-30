# Two-samples-test

## Code structure

In the simulations, what we want to estimate is the **rejection rate**. That is, run a two-sample test many times and see how often it rejects the null hypothesis. The main flow of the idea is the following:

- Generate hierarchical samples.
- Decide rejection.

So, we need methods for those two.

**Generating hierarchical samples**

To define methods for generating hierarchical samples, we first need a struct for the laws of Random Probability Measures (RPMs). This is what `structures.jl` provides: `LawRPM`, the abstract type every law of RPM subtypes, so that generator functions can accept "any law of RPM" generically. Since other methods — like computing distance, p-values, etc. — also need to work on a hierarchical sample, `structures.jl` provides a struct for that too: `HierSample`, which holds a generated hierarchical sample (an n×m matrix of atoms, plus the interval [a,b] they live in).

With those types in place, `distributions.jl` defines the concrete laws of RPM (`tnormal_normal`, `discr_normal`, `mixture`, `DP`, etc.) as subtypes of `LawRPM`, and, for each one, a `sample_exch_seq` method that draws one exchangeable sequence from it.

**Deciding rejection**

Deciding whether to reject also requires computing a distance between two samples, so the `distances/` folder provides that: `distances/hipm.jl` computes HIPM (via `dlip`), and `distances/wow.jl` computes the Wasserstein / Wasserstein-over-Wasserstein distance used by the WoW test.

`methods.jl` then combines sample generation (`distributions.jl`) and distance computation (`distances/`) to implement the tests themselves — p-values, thresholds, and reject/don't-reject decisions for HIPM, WoW, DM (Fréchet) and Energy — and exposes the `rejection_rate_*` functions that repeat the whole simulate-then-test procedure S times and report how often each test rejects. This is exactly what the `simulations/` scripts call.

For a fair comparison, all four tests run on the same hierarchical samples. That's why DM is our own Julia version of the Fréchet test of Dubey and Müller (2019), which works directly on hierarchical samples (the old R version, via `frechet::DenANOVA`, is kept commented out in `methods.jl`). For the same reason, all tests decide rejection the same way: with a permutation p-value, (1 + #{T_b ≥ T}) / (B + 1), where B is the number of permutations. The "+1" counts the observed statistic itself, which makes the test valid for any sample size.

To conclude, structures.jl, distributions.jl, hipm.jl, wow.jl, and methods.jl are foundational files for the main simulations. Now we describe the simulations.


## Running FPR/TPR simulations, Figures 2.2, 2.3

We have two scripts for estimating how often the HIPM and WoW two-sample tests reject the null hypothesis:

- **`simulations/fp.jl`** — False Positive Rate. Both samples are drawn from the *same* law of RPM, so the null hypothesis is true, and any rejection is a false positive.
- **`simulations/tp.jl`** — True Positive Rate (power). The two samples are drawn from *different* laws of RPM, so the null hypothesis is false, and a rejection is a true positive.

Both scripts share the same logic: simulate hierarchical samples many times, run the tests at a grid of significance levels θ = 0, 0.01, ..., 1, and save the estimated rejection rates to a CSV file. Each script has a detailed comment block at the top of the file (including all its command-line arguments) — this README just gives the quick-start version.

Before running either script, its simulation configuration — the law(s) of RPM Q, n, m, S, n_perm, etc — must be specified. In particular, `--label_q` in fp.jl, or `--label_q_1` and `--label_q_2` in tp.jl specify laws of RPMs. They pick one by its 1-based position from a list defined near the top of the file (`rpms` in fp.jl, `rpms_1`/`rpms_2` in tp.jl). To see which laws are available and what label maps to which, look at that list in the corresponding file.

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
3. Make sure the log folder `/home/3049277/logs/` exists (`mkdir -p /home/3049277/logs`) — SLURM doesn't create it.
4. Submit the job with `sbatch shfiles_cluster/fp.sh` (or `tp.sh`). `JULIA_NUM_THREADS` is set automatically from `--cpus-per-task`, so the simulation runs multithreaded.
5. Each task writes its own CSV into `cluster_outputs/values/` on the cluster, exactly as it would locally. Its runtime is printed at the end of its `.err` log file (`Total duration: ... seconds`).


### Plotting the results


**Important:** the plotting script reads from `values/fp_tp/`, not from `cluster_outputs/values/` where the simulation saves. If the simulation ran on the cluster, run `bash shfiles_cluster/fetch_results.sh` on your own computer: it downloads the CSVs and puts them into `values/fp_tp/`. Before replacing a file with the same name (same parameters), it moves the old one to `old_sim_values/fp_tp/<date_time>/`. It also downloads the SLURM logs (what the jobs printed, including the runtime in each `.err` file) from `~/logs/` on the cluster into `cluster_outputs/logs/`. Add `--dry-run` to only see what it would do. If the simulation ran locally, move/copy the CSVs from `cluster_outputs/values/` into `values/fp_tp/` yourself.

`fetch_results.sh` connects to the cluster under the name `cluster` and expects the project at `~/Two-samples-test` there. Set this up once on your computer: put your SSH key on the cluster (`ssh-copy-id <user>@<cluster-address>`) and add to `~/.ssh/config`:

```
Host cluster
    HostName <cluster-address>
    User <user>
```

Then `ssh cluster echo ok` should print `ok` without asking for a password.

`values_to_plots/plot_tpfp_csv.jl` reads CSV files from `values/fp_tp/` and plots HIPM vs. WoW rejection rate against θ, saving one PNG per CSV into `plots/fp_tp/`. For plotting, just run the file.

## Running mean/variance/counterexample simulations, Figures 2.4, 2.5

We have three scripts that all share the same simulation logic: estimate how often the HIPM, WoW, DM and Energy two-sample tests reject the null hypothesis as the underlying laws of RPM, Q_1 and Q_2, are made to differ. Their difference is identified by a single parameter that is swept over a grid.

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
3. Make sure the log folder `/home/3049277/logs/` exists (`mkdir -p /home/3049277/logs`) — SLURM doesn't create it.
4. Submit the job with `sbatch shfiles_cluster/mean.sh` (or `variance.sh`/`counterexample.sh`). `JULIA_NUM_THREADS` is set automatically from `--cpus-per-task`, so the simulation runs multithreaded.
5. Each task writes its own CSV into `cluster_outputs/values/` on the cluster, exactly as it would locally. Its runtime is printed at the end of its `.err` log file (`Total duration: ... seconds`).

### Plotting the results

**Important:** the plotting scripts read from `values/comparison_all/`, not from `cluster_outputs/values/` where the simulation saves. Keep only one CSV per simulation in this folder, since each plotting script plots the first file whose name matches (e.g. starts with `mean`). If the simulation ran on the cluster, run `bash shfiles_cluster/fetch_results.sh` on your own computer: it downloads the CSVs and puts them into `values/comparison_all/`, first moving all old CSVs of that simulation to `old_sim_values/comparison_all/<date_time>/`. It stops without changing anything if the cluster has more than one CSV for the same simulation. Like for FP/TP, it also downloads the SLURM logs into `cluster_outputs/logs/`, and needs the one-time SSH setup described in the FP/TP section above. Add `--dry-run` to only see what it would do. If the simulation ran locally, move/copy the CSV from `cluster_outputs/values/` into `values/comparison_all/` yourself, and move the old one out.

`values_to_plots/plot_mean.jl`, `plot_variance.jl` and `plot_counterexample.jl` each read the matching CSV from `values/comparison_all/` and plot HIPM, WoW, DM and Energy rejection rate against δ/τ/λ, saving one PNG into `plots/comparison_all/`. For plotting, just run the file.

## Rademacher threshold simulation, Figure 2.1

`simulations/section_rademacher.jl` checks how tight the theoretical Rademacher-complexity threshold for HIPM is. For fixed Q_1 = DP(1.0, Uniform(0.0, 1.0)) and Q_2 = DP(1.0, Beta(2.0, 2.0)) and θ = 0.05, it compares the empirical (1-θ)-quantile of dlip(Q_1, Q_2) (estimated by simulation) against the Rademacher threshold, across a grid of n = m, and also plots the theoretical threshold alone over a much larger grid to show its long-run decay.

### How to run locally

```bash
julia simulations/section_rademacher.jl --S 100 --n_rerun 5
```

Any argument you leave out falls back to its default value. Run with `--help` to see all options.

From VS Code / the Julia REPL (e.g. `include("simulations/section_rademacher.jl")`), no arguments are passed, so every argument uses its default value — edit the `default = ...` values inside `parse_commandline()` to change what runs.

Results are saved as two CSV files in `values/rademacher_section/`: `rademacher_emp_vs_theor_S=<S>.csv` (columns `ns`, `empirical`, `theoretical`) and `rademacher_thresholds.csv` (columns `ns`, `theoretical`). This script doesn't run on the cluster.

### Plotting the results

`values_to_plots/plot_rademacher.jl` reads both CSVs directly from `values/rademacher_section/` (no move/copy step needed) and saves the comparison plot, the solo decay plot, and both combined side by side to `plots/rademacher_section/`. For plotting, just run the file.

## Mortality dataset analysis

Besides the simulations, `mortality/` applies the tests to real data from the Human Mortality Database: for each year 1960–2010 and each gender, it tests whether Soviet-Union-influenced countries and the rest have the same distribution of age-at-death.

- **`mortality/data_extractors.jl`** — loads the per-country life tables from `mortality/dataset/<gender>/` and turns them into the hierarchical samples / PMFs the other two scripts need. It's a library, not something you run directly.
- **`mortality/eda.jl`** — exploratory data analysis plots (data lost under age truncation, individual country PMFs, PMF/quantile/pooled-PMF comparisons between the two groups). Figure 2.6.
- **`mortality/pvalues.jl`** — the actual test: computes p-values via HIPM, WoW, Fréchet, and two baselines (averaging, pooling) for each year/gender. For Fréchet, each country's age-at-death distribution is first turned into its exact quantile function, so it can be used like a row of a hierarchical sample and the Wasserstein distances are computed exactly.



`pvalues.jl` saves one file per gender/method to `values/mortality_dataset/pvalues_<gender>_<method>.txt`. `values_to_plots/plot_pvalues.jl` reads directly from that folder (everything here runs locally, so there's no cluster move/copy step) and saves five PNGs per gender to `plots/mortality_dataset/`: `pvalues_<gender>_hipm_wow_av_pool.png` (HIPM, WoW, averaging, pooling), `pvalues_<gender>_hipm_wow_frechet.png` (HIPM, WoW, Fréchet), and `pvalues_<gender>_frechet.png`, `pvalues_<gender>_hipm.png`, `pvalues_<gender>_wow.png` (one method alone). The two plots with Fréchet have a higher y-axis limit (0.4 instead of 0.3), since Fréchet p-values go up to about 0.35. This corresponds to Figure 2.7.

## Making all plots at once

Instead of running each plotting script by hand, run:

```bash
julia --project=. values_to_plots/plot_all.jl
```

(or `include("values_to_plots/plot_all.jl")` from VS Code / the Julia REPL). It runs every `plot_*.jl` file in `values_to_plots/`, so each plot is saved into its usual folder in `plots/`. If one script fails, for example because its CSV isn't in `values/` yet, the others still run and the failed ones are listed at the end. When a new plotting script is added, add its name to `PLOT_SCRIPTS` at the top of `plot_all.jl`.

## Checking correctness

Before trusting the simulations, we want to know that the tests are computed correctly. `check_correctness_time_hipm_wow/` has one script per test:

- **`testhipm.jl`**, **`testwow.jl`** — compare our HIPM and WoW distances against Hugo & Marta's original code, and time both versions.
- **`testfrechet.jl`** — compares our Fréchet test statistic against the R package `frechet` (Dubey and Müller's own code) on simulated hierarchical samples, and checks errors on bad inputs, p-values and rejection rates. It needs R with the `frechet` package installed.

Run them from the project root, e.g.:

```bash
julia --project=. check_correctness_time_hipm_wow/testfrechet.jl
```

## Cluster run times

Cluster times for current parameters:

| Simulation | Runtime |
| --- | --- |
| `counterexample` | 47h 31m |
| `variance` | 69h 47m |
| `mean` | 81h 13m |
| `tp_4_4` | 3h 51m |
| `fp_4` | 4h 1m |
