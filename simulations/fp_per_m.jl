using DataFrames
using ArgParse
using ExtractMacro
using CSV




include(joinpath(pwd(),"methods.jl"))

# In this file we run simulations to estimate the false positive rate. In particular
# we fix θ = 0.05 and report FPR per each m ∈ {10, 20, 30, 40, 50}

# The list of random probability meaures we use in the simulations are the following:
rpms = [beta_beta_A(1.0, 1.0), 
        normal_normal_A(0.0, 1.0), 
        DP(1.0, Uniform(0,1)),
        DP(1.0, Normal(0.0, 1.0))]

names_of_rpms = ["label=1", 
                "label=2",
                "label=3",
                "label=4"]

# Julia always provides the global variable ARGS.
# When this file is run from the terminal, ARGS contains things writen next to the run command,
# which override the default values.
# When this file is run from VS Code or via include(), ARGS is empty,
# so the default values are used.
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


struct SimulationConfig
    Q::LawRPM
    n::Int
    S::Int
    n_perm::Int
end


function run_simulation(config::SimulationConfig)
    @extract config : Q n S n_perm
    θ = 0.05
    ms = [10, 20, 30, 40, 50]
    fp_hipm = zeros(length(ms))
    fp_wow = zeros(length(ms))

    for (i, m) in enumerate(ms)
        fp_hipm[i], fp_wow[i] = rejection_rate_hipm_wow(Q, Q, n, m, S, [θ], n_perm, false)
    end
    return DataFrame(ms = ms, fp_hipm = fp_hipm, fp_wow = fp_wow)
end

function main()
    parsed_args = parse_commandline()
    label_q = parsed_args["label_q"]
    Q = rpms[label_q]
    n, S, n_perm = parsed_args["n"], parsed_args["S"], parsed_args["n_perm"]
    config = SimulationConfig(Q, n, S, n_perm)
    df = run_simulation(config)

    file_name = "fp_per_m_"*names_of_rpms[label_q]*"_"*"n=$(n)_S=$(S)_n_perm=$(n_perm)"
    file_path = joinpath(pwd(), "cluster_outputs", "values")
    mkpath(file_path) # create folder if it's missing
    file_path = joinpath(file_path, file_name * ".csv")
    CSV.write(file_path, df)
    @info "Dataframe saved successfully" path=file_path
end

@info "number of threads: $(Threads.nthreads())"
t_start = time()
main()
@info "Total duration: $(round(time() - t_start, digits=2)) seconds"


