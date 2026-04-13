using DataFrames
using ArgParse
using ExtractMacro
using CSV



# In this file we run simulations to estimate the false positive rate. In particular
# we fix θ = 0.05 and report FPR per each m ∈ {10, 20, 30, 40, 50}
include(joinpath(pwd(),"methods.jl"))



function rejection_rate_hipm_wow_per_m(
    q_1::LawRPM,
    q_2::LawRPM,
    n::Int,
    m::Vector{Int},
    S::Int,
    θ::Vector{Float64},
    n_samples::Int,
    bootstrap::Bool
)

    rates_hipm = zeros(length(m))
    rates_wow = zeros(length(m))

    @floop ThreadedEx() for s in 1:S
        h_1 = generate_hiersample(q_1, n, m[end])
        h_2 = generate_hiersample(q_2, n, m[end])

        local_hipm = zeros(length(m))
        local_wow = zeros(length(m))

        for i in eachindex(m)
            if i == length(m)
                local_hipm[i] = decision_hipm(h_1, h_2, θ, n_samples, bootstrap)[1]
                local_wow[i] = decision_wow(h_1, h_2, θ, n_samples, bootstrap)[1]
            else
                h_1_m = HierSample(h_1.atoms[:, 1:m[i]], h_1.a, h_1.b)
                h_2_m = HierSample(h_2.atoms[:, 1:m[i]], h_2.a, h_2.b)

                local_hipm[i] = decision_hipm(h_1_m, h_2_m, θ, n_samples, bootstrap)[1]
                local_wow[i] = decision_wow(h_1_m, h_2_m, θ, n_samples, bootstrap)[1]
            end
        end

        @reduce rates_hipm .+= local_hipm
        @reduce rates_wow .+= local_wow
    end

    rates_hipm ./= S
    rates_wow ./= S

    return rates_hipm, rates_wow
end



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
    n_repeat = 10
    results = DataFrame(repeat = Int[], m = Int[], fp_hipm = Float64[], fp_wow = Float64[])
    for r in 1:n_repeat
        fp_hipm_per_m, fp_wow_per_m = rejection_rate_hipm_wow_per_m(Q, Q, n, ms, S, [θ], n_perm, false)
        append!(results, DataFrame(
        repeat = fill(r, length(ms)),
        m = ms,
        fp_hipm = fp_hipm_per_m,
        fp_wow = fp_wow_per_m
        ))
    end
    return results
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


