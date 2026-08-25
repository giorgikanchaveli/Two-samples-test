# =============================================================================
# fp.jl — False Positive Rate simulations
# =============================================================================
#
# Estimates the false positive rate (Type I error) using HIPM and WoW. Samples are generated from the SAME law of random
# probability measure Q, so the null hypothesis is always true.
#

# The flow of the simulation is the following: 
# 1) Simulation configurations (law of RPM Q, n, m, ...) must be specified.
# 2) Run method to estimate rejection rate given simulation configurations.
# 3) Save simulation values in the folder <project_root>/cluster_outputs/values/.




# HOW TO RUN:
#   • From a terminal, pass arguments after the file name, e.g.:
#       julia fp.jl --label_q 3 --n 20 --m 30 --S 500 --n_perm 200
#     Any argument left out falls back to its default (see parse_commandline
#     below). 

#   • From VS Code / the Julia REPL / `include("fp.jl")`: ALL arguments get their default values. So if we want to change them,  we should edit the default values. Then just run the file.







# Output: a CSV file with columns θs, fp_hipm, fp_wow, saved to
#   <project_root>/cluster_outputs/values/fp_label=<label_q>_n=<n>_m=<m>_S=<S>_n_perm=<n_perm>.csv
# (the folder is created automatically if it doesn't already exist). Per each value of θ, FPR is saved for HIPM and WoW.

# To plot these results, run the file values_to_plots/plot_tpfp_csv.jl — it reads the
# saved dataframes, plots fp_hipm and fp_wow against θs and saves the plot in the folder <project_root>/plots/fp_tp
# Note: plot_tpfp_csv.jl expects CSVs in <project_root>/values/fp_tp, not cluster_outputs/values.
# Since simulations are run on the cluster, manually move/copy the CSVs from cluster_outputs/values
# into values/fp_tp before running plot_tpfp_csv.jl.





using DataFrames
using ArgParse
using ExtractMacro
using CSV

include(joinpath(pwd(),"methods.jl")) # brings in useful simulation functions. 

# Laws of RPMs available for the simulation.
# --label_q selects one of these by its 1-based position in the list
# (e.g. label_q = 3 runs the simulation with Q = DP(1.0, Uniform(0,1))).
rpms = [beta_beta_A(1.0, 1.0),
        normal_normal_A(0.0, 1.0),
        DP(1.0, Uniform(0,1)),
        DP(1.0, Normal(0.0, 1.0)),
        normal_normal_B(0.0, 10.0, 10.0),
        normal_normal_B(0.0, 50.0, 50.0)]

# Human-readable tag for each entry in `rpms` above, used only to build the output CSV's filename (see main() below).
names_of_rpms = ["label=1",
                "label=2",
                "label=3",
                "label=4",
                "label=5",
                "label=6"]



# Defines the command-line arguments (--label_q, --n, --m, --S, --n_perm) accepted by this file,
# together with their help text and default values (used when no arguments are passed, e.g. from VS Code).
function parse_commandline()

    s = ArgParseSettings(description = "Run Simulations for FP,TP,ROC")

    @add_arg_table! s begin
        "--label_q"
            help = "label for law of RPM"
            arg_type = Int
            default = 3
        "--n"
            help = "number of rows n"
            arg_type = Int
            default = 1
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




# Bundles everything run_simulation needs: the law of RPM Q,
# the sample sizes (n exchangeable sequences of length m), the number of MCMC
# repetitions (S), and the number of permutation samples (n_perm).
struct SimulationConfig
    Q::LawRPM
    n::Int
    m::Int
    S::Int
    n_perm::Int
end


function run_simulation(config::SimulationConfig)
    @extract config : Q n m S n_perm # unpacks the struct's fields into local variables Q, n, m, S, n_perm
    θs = collect(0.0:0.01:1.0) # grid of significance levels at which the rejection rate is evaluated

    fp_hipm, fp_wow = rejection_rate_hipm_wow(Q, Q, n, m, S, θs, n_perm, false) # estimates FPR using permutation approach.
    return DataFrame(θs = θs, fp_hipm = fp_hipm, fp_wow = fp_wow) 
end

function main()
    parsed_args = parse_commandline()
    label_q = parsed_args["label_q"]
    Q = rpms[label_q] # law of RPM
    n, m, S, n_perm = parsed_args["n"], parsed_args["m"], parsed_args["S"], parsed_args["n_perm"]
    config = SimulationConfig(Q, n, m, S, n_perm)  

    df = run_simulation(config)

    # Result is saved to:
    #   <project_root>/cluster_outputs/values/fp_label=<label_q>_n=<n>_m=<m>_S=<S>_n_perm=<n_perm>.csv
    file_name = "fp_"*names_of_rpms[label_q]*"_"*"n=$(n)_m=$(m)_S=$(S)_n_perm=$(n_perm)"
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


