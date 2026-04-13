using DataFrames
using ArgParse
using ExtractMacro
using CSV




include(joinpath(pwd(),"methods.jl"))

# In this file we run simulations to estimate the false positive rate.

# The list of random probability meaures we use in the simulations are the following:
rpms = [beta_beta_A(1.0, 1.0), 
        normal_normal_A(0.0, 1.0), 
        DP(1.0, Uniform(0,1)),
        DP(1.0, Normal(0.0, 1.0)),
        normal_normal_B(0.0, 10.0, 10.0),
        normal_normal_B(0.0, 50.0, 50.0)]

names_of_rpms = ["label=1", 
                "label=2",
                "label=3",
                "label=4",
                "label=5",
                "label=6"]

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


struct SimulationConfig
    Q::LawRPM
    n::Int
    m::Int
    S::Int
    n_perm::Int
end


function run_simulation(config::SimulationConfig)
    @extract config : Q n m S n_perm
    θs = collect(0.0:0.01:1.0)
    fp_hipm, fp_wow = rejection_rate_hipm_wow(Q, Q, n, m, S, θs, n_perm, false)
    return DataFrame(θs = θs, fp_hipm = fp_hipm, fp_wow = fp_wow)
end

function main()
    parsed_args = parse_commandline()
    label_q = parsed_args["label_q"]
    Q = rpms[label_q]
    n, m, S, n_perm = parsed_args["n"], parsed_args["m"], parsed_args["S"], parsed_args["n_perm"]
    config = SimulationConfig(Q, n, m, S, n_perm)
    df = run_simulation(config)

    file_name = "fp_"*names_of_rpms[label_q]*"_"*"n=$(n)_m=$(m)_S=$(S)_n_perm=$(n_perm)"
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


