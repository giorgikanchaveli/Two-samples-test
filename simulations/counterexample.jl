# =============================================================================
# counterexample.jl — Rejection rate simulations
# =============================================================================
#
# Estimates the rejection rate of four testing schemes (HIPM, WoW, DM, Energy) for two samples drawn
# from discr_normal / mixture laws of RPM (see distributions.jl). Q_1 = discr_normal([-1.0, 1.0], [0.5, 0.5])
# is a law of RPM under which the random probability measure is Gaussian(δ, 1), where δ equals -1.0 or
# 1.0 with equal probability.

# In this simulations, Q_2 = mixture(Q_1, Q_aux, λ) = λ*Q_1 + (1-λ)*Q_aux, where
# Q_aux = discr_normal([-2.0, 0.0, 2.0], [1/8, 3/4, 1/8]) and λ
# ranges over a grid in [0.0, 1.0] — so Q_1 and Q_2 differ only in λ. At λ = 1 the null hypothesis is
# true (Q_2 = Q_1); at λ = 0, Q_2 = Q_aux and the two laws differ the most.
#
# The flow of the simulation is the following:
# 1) Simulation configurations (n, m, S, n_samples, θ, bootstrap) must be specified.
# 2) Run method to estimate rejection rate for each λ in the grid.
# 3) Save simulation values in the folder <project_root>/cluster_outputs/values/.

# HOW TO RUN:
#   • From a terminal, pass arguments after the file name, e.g.:
#       julia counterexample.jl --n 20 --m 30 --S 500 --n_samples 200
#     Any argument left out falls back to its default (see parse_commandline
#     below).

#   • From VS Code / the Julia REPL / `include("counterexample.jl")`: ALL arguments get their default values.
#     So if we want to change them, we should edit the default values. θ and bootstrap are set
#     directly in main() rather than via command line, so edit them there if needed. Then just run the file.

# Output: a CSV file with columns λs, hipm, wow, dm, energy, saved to
#   <project_root>/cluster_outputs/values/counterexample_n=<n>_m=<m>_S=<S>_bootstrap=<bootstrap>_n_samples=<n_samples>.csv
# (the folder is created automatically if it doesn't already exist). Per each value of λ, the
# rejection rate is saved for HIPM, WoW, DM and Energy.

# To plot these results, run the file values_to_plots/plot_counterexample.jl — it reads the saved dataframe
# (CSV in the folder whose name starts with "counterexample"), plots hipm, wow, dm and energy
# against λs, and saves the plot in the folder <project_root>/plots/comparison_all.

# Important Note: plot_counterexample.jl expects CSVs in <project_root>/values/comparison_all, not cluster_outputs/values.
# Since simulations are run on the cluster, manually move/copy the CSVs from cluster_outputs/values
# into values/comparison_all before running plot_counterexample.jl.

using DataFrames
using ArgParse
using ExtractMacro
using CSV

include(joinpath(pwd(),"methods.jl")) # brings in useful simulation functions.

# Defines the command-line arguments (--n, --m, --S, --n_samples) accepted by this file,
# together with their help text and default values (used when no arguments are passed, e.g. from VS Code).
function parse_commandline()

    s = ArgParseSettings(description = "Run Simulations for counterexample")

    @add_arg_table! s begin
        "--n"
            help = "number of rows n"
            arg_type = Int
            default = 2
        "--m"
            help = "number of columns m"
            arg_type = Int
            default = 2
        "--S"
            help = "Number of MCMC iterations S"
            arg_type = Int
            default = 1
        "--n_samples"
            help = "Number of bootstrap/permutation samples n_samples"
            arg_type = Int
            default = 1
    end
    return parse_args(s)
end


# Bundles everything run_simulation needs: the sample sizes (n exchangeable sequences of length m),
# the number of MCMC repetitions (S), the number of bootstrap/permutation samples (n_samples),
# the significance level θ, and whether to use the bootstrap or permutation approach (bootstrap).
struct SimConfig
    n::Int
    m::Int
    S::Int
    n_samples::Int
    θ::Float64
    bootstrap::Bool
end


function run_simulation(config::SimConfig)
    λs = collect(0.0:0.05:1.0) # grid of λ values used to build Q_2 as a mixture of q_1 and q_2_aux
    q_1 = discr_normal([-1.0, 1.0], [0.5, 0.5])
    q_2_aux = discr_normal([-2.0, 0.0, 2.0], [1/8, 3/4, 1/8])
    pairs = [(q_1, mixture(q_1, q_2_aux, λ)) for λ in λs] # Q_1 fixed, Q_2 = λ*q_1 + (1-λ)*q_2_aux
    total_sims = length(pairs)

    @extract config : n m S n_samples θ bootstrap # unpacks the struct's fields into local variables n, m, S, n_samples, θ, bootstrap

    @info "Starting simulation for counterexample with parameters: n = $n, m = $m, S = $S, n_samples = $(n_samples)."
    results = map(enumerate(pairs)) do (i, p) # per each pair collect the results
        # rejection_rate_all returns (hipm, wow, dm, energy)
        q_1,q_2 = p
        out = rejection_rate_all(q_1, q_2, n, m, S, θ, n_samples, bootstrap)
        @info "Progress $i / $(total_sims)"
        return out
    end
    return DataFrame(λs = λs,
                     hipm = [r[1] for r in results],
                     wow = [r[2] for r in results],
                     dm = [r[3] for r in results],
                     energy = [r[4] for r in results])
end


function save_results(df::DataFrame, config::SimConfig)

    @extract config : n m S n_samples θ bootstrap

    values_path = joinpath(pwd(), "cluster_outputs", "values")
    mkpath(values_path) # create folder if it's missing

    file_name = "counterexample_n=$(n)_m=$(m)_S=$(S)_bootstrap=$(bootstrap)_n_samples=$(n_samples)"

    values_path = joinpath(values_path, file_name * ".csv")
    CSV.write(values_path, df)
    @info "Dataframe saved successfully" path=values_path
end


function main()
    θ = 0.05
    bootstrap = false
    parsed_args = parse_commandline()

    config = SimConfig(
        parsed_args["n"],
        parsed_args["m"],
        parsed_args["S"],
        parsed_args["n_samples"],
        θ, bootstrap
        )

    df = run_simulation(config)
    save_results(df, config)

end

@info "number of threads: $(Threads.nthreads())"
t_start = time()
main() # runs the simulation and writes the CSV described above
@info "Total duration: $(round(time() - t_start, digits=2)) seconds"