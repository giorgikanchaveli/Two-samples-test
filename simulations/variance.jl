# =============================================================================
# variance.jl — Rejection rate simulations
# =============================================================================
#
# Estimates the rejection rate of four testing schemes (HIPM, WoW, DM, Energy) for two samples drawn
# from tnormal_normal laws of RPM (see distributions.jl). tnormal_normal(δ, σ, a, b) defines a law of
# RPM Q under which the random probability measure is Gaussian(μ, 1), where μ
# is itself random, drawn from a Truncated Normal on [a, b] with mean δ and standard deviation σ.

# In this simulations, Q_1 = tnormal_normal(0.0, 0.2, -10.0, 10.0) and
# Q_2 = tnormal_normal(0.0, 0.2*τ, -10.0, 10.0), where τ
# ranges over a grid in [0.1, 2.5] — so Q_1 and Q_2 differ only in their standard deviation σ, scaled by τ.
# At τ = 1 the null hypothesis is true (Q_1 = Q_2); for τ ≠ 1 it's false.
#
# The flow of the simulation is the following:
# 1) Simulation configurations (n, m, S, n_samples, θ, bootstrap) must be specified.
# 2) Run method to estimate rejection rate for each τ in the grid.
# 3) Save simulation values in the folder <project_root>/cluster_outputs/values/.

# HOW TO RUN:
#   • From a terminal, pass arguments after the file name, e.g.:
#       julia variance.jl --n 20 --m 30 --S 500 --n_samples 200
#     Any argument left out falls back to its default (see parse_commandline
#     below).

#   • From VS Code / the Julia REPL / `include("variance.jl")`: ALL arguments get their default values.
#     So if we want to change them, we should edit the default values. θ and bootstrap are set
#     directly in main() rather than via command line, so edit them there if needed. Then just run the file.

# Output: a CSV file with columns τs, hipm, wow, dm, energy, saved to
#   <project_root>/cluster_outputs/values/variance_n=<n>_m=<m>_S=<S>_bootstrap=<bootstrap>_n_samples=<n_samples>.csv
# (the folder is created automatically if it doesn't already exist). Per each value of τ, the
# rejection rate is saved for HIPM, WoW, DM and Energy.

# To plot these results, run the file values_to_plots/plot_variance.jl — it reads the saved dataframe
# (CSV in the folder whose name starts with "variance"), plots hipm, wow, dm and energy
# against τs, and saves the plot in the folder <project_root>/plots/comparison_all.

# Important Note: plot_variance.jl expects CSVs in <project_root>/values/comparison_all, not cluster_outputs/values.
# Since simulations are run on the cluster, manually move/copy the CSVs from cluster_outputs/values
# into values/comparison_all before running plot_variance.jl.

using DataFrames
using ArgParse
using ExtractMacro
using CSV

include(joinpath(pwd(),"methods.jl")) # brings in useful simulation functions.

# Defines the command-line arguments (--n, --m, --S, --n_samples) accepted by this file,
# together with their help text and default values (used when no arguments are passed, e.g. from VS Code).
function parse_commandline()

    s = ArgParseSettings(description = "Run Simulations for variance")

    @add_arg_table! s begin
        "--n"
            help = "number of rows n"
            arg_type = Int
            default = 100
        "--m"
            help = "number of columns m"
            arg_type = Int
            default = 100
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
    τs = collect(0.1:0.05:2.5) # grid of τ values used to scale Q_2's standard deviation
    pairs = [(tnormal_normal(0.0,0.2,-10.0,10.0), tnormal_normal(0.0,0.2*τ,-10.0,10.0)) for τ in τs] # Q_1 fixed at σ=0.2, Q_2 scaled by τ
    total_sims = length(pairs)

    @extract config : n m S n_samples θ bootstrap # unpacks the struct's fields into local variables n, m, S, n_samples, θ, bootstrap

    @info "Starting simulation for variance with parameters: n = $n, m = $m, S = $S, n_samples = $(n_samples)."
    results = map(enumerate(pairs)) do (i, p) # per each pair collect the results
        # rejection_rate_all returns (hipm, wow, dm, energy)
        q_1,q_2 = p
        out = rejection_rate_all(q_1, q_2, n, m, S, θ, n_samples, bootstrap)
        @info "Progress $i / $(total_sims)"
        return out
    end
    return DataFrame(τs = τs,
                     hipm = [r[1] for r in results],
                     wow = [r[2] for r in results],
                     dm = [r[3] for r in results],
                     energy = [r[4] for r in results])
end


function save_results(df::DataFrame, config::SimConfig)

    @extract config : n m S n_samples θ bootstrap

    values_path = joinpath(pwd(), "cluster_outputs", "values")
    mkpath(values_path) # create folder if it's missing

    file_name = "variance_n=$(n)_m=$(m)_S=$(S)_bootstrap=$(bootstrap)_n_samples=$(n_samples)"

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