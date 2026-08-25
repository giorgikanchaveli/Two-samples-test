# =============================================================================
# tp.jl — True Positive Rate simulations
# =============================================================================
#
# Estimates the true positive rate (power) using HIPM and WoW. Samples are generated from two DIFFERENT laws of random
# probability measure Q_1 and Q_2, so the null hypothesis is always false.
#
# The flow of the simulation is the following:
# 1) Simulation configurations (laws of RPM Q_1, Q_2, n, m, ...) must be specified.
# 2) Run method to estimate rejection rate given simulation configurations.
# 3) Save simulation values in the folder <project_root>/cluster_outputs/values/.

# HOW TO RUN:
#   • From a terminal, pass arguments after the file name, e.g.:
#       julia tp.jl --label_q_1 1 --label_q_2 1 --n 20 --m 30 --S 500 --n_perm 200
#     Any argument left out falls back to its default (see parse_commandline
#     below).

#   • From VS Code / the Julia REPL / `include("tp.jl")`: ALL arguments get their default values. So if we want to change them, we should edit the default values. Then just run the file.

# Output: a CSV file with columns θs, tp_hipm, tp_wow, saved to
#   <project_root>/cluster_outputs/values/tp_label_1=<label_q_1>_label_2=<label_q_2>_n=<n>_m=<m>_S=<S>_n_perm=<n_perm>.csv
# (the folder is created automatically if it doesn't already exist). Per each value of θ, TPR (power) is saved for HIPM and WoW.

# To plot these results, run the file values_to_plots/plot_tpfp_csv.jl — it reads the
# saved dataframes, plots tp_hipm and tp_wow against θs and saves the plot in the folder <project_root>/plots/fp_tp
# Note: plot_tpfp_csv.jl expects CSVs in <project_root>/values/fp_tp, not cluster_outputs/values.
# Since simulations are run on the cluster, manually move/copy the CSVs from cluster_outputs/values
# into values/fp_tp before running plot_tpfp_csv.jl.


using DataFrames
using ArgParse
using ExtractMacro
using CSV

include(joinpath(pwd(),"methods.jl")) # brings in useful simulation functions.

# Laws of RPMs used as the first sample (Q_1). --label_q_1 (see in parse_commandline()) selects one of these by its 1-based position in the list.
rpms_1 = [beta_beta_A(1.0, 1.0),
        normal_normal_A(0.0, 1.0),
        DP(1.0, Uniform(0,1)),
        DP(1.0, Normal(0.0, 1.0))
       ]

# Laws of RPMs used as the second sample (Q_2). --label_q_2 (see in parse_commandline()) selects one of these by its 1-based position in the list.

rpms_2 = [beta_beta_A(1.0, 1.5),
          normal_normal_A(0.0, 1.5),
          DP(1.0, Beta(1.0,1.4)),
          DP(2.0, Normal(0.0, 1.0))
          ]


# Defines the command-line arguments (--label_q_1, --label_q_2, --n, --m, --S, --n_perm) accepted by this file,
# together with their help text and default values (used when no arguments are passed, e.g. from VS Code).
function parse_commandline()

    s = ArgParseSettings(description = "Run Simulations for tp,TP,ROC")

    @add_arg_table! s begin
        "--label_q_1"
            help = "label for first law of RPM"
            arg_type = Int
            default = 1
        "--label_q_2"
            help = "label for second law of RPM"
            arg_type = Int
            default = 1
        "--n"
            help = "number of rows n"
            arg_type = Int
            default = 3
        "--m"
            help = "number of columns m"
            arg_type = Int
            default = 1
        "--S"
            help = "number of mcmc iterations"
            arg_type = Int
            default = 1
        "--n_perm"
            help = "number of samples via permutation approach"
            arg_type = Int
            default = 1
    end
    return parse_args(s)
end


# Bundles everything run_simulation needs: the two laws of RPM Q_1, Q_2,
# the sample sizes (n exchangeable sequences of length m), the number of MCMC
# repetitions (S), and the number of permutation samples (n_perm).
struct SimulationConfig
    Q_1::LawRPM
    Q_2::LawRPM
    n::Int
    m::Int
    S::Int
    n_perm::Int
end


function run_simulation(config::SimulationConfig)
    @extract config : Q_1 Q_2 n m S n_perm # unpacks the struct's fields into local variables Q_1, Q_2, n, m, S, n_perm
    θs = collect(0.0:0.01:1.0) # grid of significance levels at which the rejection rate is evaluated

    tp_hipm, tp_wow = rejection_rate_hipm_wow(Q_1, Q_2, n, m, S, θs, n_perm, false) # estimates TPR using permutation approach.
    return DataFrame(θs = θs, tp_hipm = tp_hipm, tp_wow = tp_wow)
end

function main()
    parsed_args = parse_commandline()
    label_q_1 = parsed_args["label_q_1"]
    label_q_2 = parsed_args["label_q_2"]
    Q_1 = rpms_1[label_q_1] # first law of RPM
    Q_2 = rpms_2[label_q_2] # second law of RPM
    n, m, S, n_perm = parsed_args["n"], parsed_args["m"], parsed_args["S"], parsed_args["n_perm"]
    config = SimulationConfig(Q_1, Q_2, n, m, S, n_perm)

    df = run_simulation(config)

    # Result is saved to:
    #   <project_root>/cluster_outputs/values/tp_label_1=<label_q_1>_label_2=<label_q_2>_n=<n>_m=<m>_S=<S>_n_perm=<n_perm>.csv
    file_name = "tp_label_1=$(label_q_1)_label_2=$(label_q_2)_n=$(n)_m=$(m)_S=$(S)_n_perm=$(n_perm)"
    file_path = joinpath(pwd(), "cluster_outputs", "values")
    mkpath(file_path) # create folder if it's missing
    file_path = joinpath(file_path, file_name * ".csv")
    CSV.write(file_path, df)
    @info "Dataframe saved successfully" path=file_path
end

@info "number of threads: $(Threads.nthreads())"
t_start = time()
main() # runs the simulation and writes the CSV described above
@info "Total duration: $(round(time() - t_start, digits=2)) seconds"


