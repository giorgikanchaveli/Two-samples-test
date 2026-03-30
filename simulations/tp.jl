using DataFrames
using ArgParse
using ExtractMacro
using CSV



# In this file we run simulations to estimate the True positive rate.

include(joinpath(pwd(),"methods.jl"))


# In the following we write list of RPMs. 
rpms_1 = [beta_beta_A(1.0, 1.0), 
        normal_normal_A(0.0, 1.0), 
        DP(1.0, Uniform(0,1)),
        DP(1.0, Normal(0.0, 1.0))]


rpms_2 = [beta_beta_B(1.0, 1.0),
          beta_beta_A(2.0, 1.0),
          normal_normal_B(0.0, 1.0, sqrt(2)),
          normal_normal_A(1.0, 1.0),
          DP(2.0, Uniform(0.0, 1.0)),
          DP(1.0, Beta(2.0,2.0)),
          DP(2.0, Normal(0.0, 1.0)),
          DP(1.0, Normal(1.0, 1.0))]

    

# Julia always provides the global variable ARGS.
# When this file is run from the terminal, ARGS contains things writen next to the run command,
# which override the default values.
# When this file is run from VS Code or via include(), ARGS is empty,
# so the default values are used.
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
            default = 2
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
    Q_1::LawRPM
    Q_2::LawRPM
    n::Int
    m::Int
    S::Int
    n_perm::Int
end


function run_simulation(config::SimulationConfig)
    @extract config : Q_1 Q_2 n m S n_perm
    θs = collect(0.0:0.01:1.0)
    tp_hipm, tp_wow = rejection_rate_hipm_wow(Q_1, Q_2, n, m, S, θs, n_perm, false)
    return DataFrame(θs = θs, tp_hipm = tp_hipm, tp_wow = tp_wow)
end

function main()
    parsed_args = parse_commandline()
    label_q_1 = parsed_args["label_q_1"]
    label_q_2 = parsed_args["label_q_2"]
    Q_1 = rpms_1[label_q_1]
    Q_2 = rpms_2[label_q_2]
    n, m, S, n_perm = parsed_args["n"], parsed_args["m"], parsed_args["S"], parsed_args["n_perm"]
    config = SimulationConfig(Q_1, Q_2, n, m, S, n_perm)
    df = run_simulation(config)

    file_name = "tp_label_1=$(label_q_1)_label_2=$(label_q_2)_n=$(n)_m=$(m)_S=$(S)_n_perm=$(n_perm)"
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


