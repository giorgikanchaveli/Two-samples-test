using Plots
using DataFrames
using ArgParse
using ExtractMacro



include(joinpath(pwd(),"methods.jl"))

# Julia always provides the global variable ARGS.
# When this file is run from the terminal, ARGS contains the command-line arguments,
# which override the default values.
# When this file is run from VS Code or via include(), ARGS is empty,
# so the default values are used.
function parse_commandline()
    
    s = ArgParseSettings(description = "Run Simulations for Figure 1, mean")

    @add_arg_table! s begin
        "--n"
            help = "number of rows n"
            arg_type = Int
            default = 1
        "--m"
            help = "number of columns m"
            arg_type = Int
            default = 4
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


# All the parameters for simulations.
struct SimConfig
    n::Int
    m::Int
    S::Int
    n_samples::Int
    θ::Float64
    bootstrap::Bool
end


function run_simulation(config::SimConfig)
    τs = collect(0.1:0.05:2.5)
    pairs = [(tnormal_normal(0.0,0.2,-10.0,10.0), tnormal_normal(0.0,0.2*τ,-10.0,10.0)) for τ in τs]
    total_sims = length(pairs)

    @extract config : n m S n_samples θ bootstrap
    
    @info "Starting simulation for variance with parameters: n = $n, m = $m, S = $S, n_samples = $(n_samples)."
    results = map(enumerate(pairs)) do (i, p) # per each pair collect the results
        # Assuming rejection_rate_all returns (hipm, wow, dm, energy)
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
    
    output_dir = "plotscluster"
    mkpath(output_dir) # create folder if it's missing

  
   
    file_name = "variance_n=$(n)_m=$(m)_S=$(S)_bootstrap=$(bootstrap)_n_samples=$(n_samples)"

    fig = plot(
            title = "Rejection rates for 4 schemes",
            xlabel = "τ",
            ylabel = "Rej rate",
            xlims=(minimum(df.τs) - 0.05, maximum(df.τs)+ 0.05),
            ylims = (-0.1, 1.1))
    plot!(fig, df.τs, df.dm, label = "DM", color = "red")
    plot!(fig, df.τs, df.energy, label = "Energy", color = "blue")
    plot!(fig, df.τs, df.hipm, label = "HIPM", color = "green")
    plot!(fig, df.τs, df.wow, label = "WoW", color = "brown")
    
    full_path = joinpath(output_dir, file_name * ".png")
    savefig(fig, full_path)
    # csv_path  = joinpath(output_dir, file_base * ".csv") # to save also returned values
    @info "Plot saved successfully" path=full_path
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
main()
@info "Total duration: $(round(time() - t_start, digits=2)) seconds"

































# using Plots



# include("methods.jl")



# function save_fig(pairs::Vector{<:Tuple{LawRPM,LawRPM}}, param_pairs::Vector{Float64}, file_name::String, file_path::String, title::String, xlabel::String, ylabel::String,
#     n::Int, m::Int, S::Int, θ::Float64, n_samples::Int, bootstrap::Bool)
#     rates_hipm = zeros(length(param_pairs))
#     rates_wow = zeros(length(param_pairs))
#     rates_dm = zeros(length(param_pairs))
#     rates_energy = zeros(length(param_pairs))
#     for i in 1:length(pairs)
#         q_1, q_2 = pairs[i]
#         r_hipm, r_wow, r_dm, r_energy = rejection_rate_all(q_1,q_2,n,m,S,θ,n_samples,bootstrap)
#         rates_hipm[i] = r_hipm
#         rates_wow[i] = r_wow
#         rates_dm[i] = r_dm
#         rates_energy[i] = r_energy
#         println(i)
#     end
#     fig = plot(title = title, xlabel = xlabel, ylabel = ylabel, xlims=(minimum(param_pairs) - 0.05, maximum(param_pairs)+ 0.05),
#                          ylims = (-0.1, 1.1))
#     plot!(fig, param_pairs, rates_dm, label = "dm", color = "red", marker = (:circle, 4))
#     plot!(fig, param_pairs, rates_hipm, label = "hipm", color = "green", marker = (:circle, 4))
#     plot!(fig, param_pairs, rates_wow, label = "wow", color = "brown", marker = (:circle, 4))
#     plot!(fig, param_pairs, rates_energy, label = "Energy", color = "blue", marker = (:circle, 4))
#     filepath = joinpath(pwd(), file_path)
#     savefig(fig,joinpath(filepath, file_name))
# end



# # obtain times: for S = 4, 1 pair of RPMS, n = 100, m = 100 and n_samples = 100.

# # wow : 23 seconds (40 minutes in total if proper threshold, S = 400)
# #        70 seconds in total if wrong threshold, S = 400
        
# # hipm : 50 seconds ( 83 minutes in total if proper threshold, S = 400)
# #         210 seconds in total if wrong threshold, S = 400    

# # energy: 4 seconds 
# #           400 seconds in total, S = 400

# # dm : 0.7 seconds ( S = 1 )
# #      280 seconds in total, S = 400

# # total S = 400 for one pair is 960 seconds


# println("running file fig_1_variance.jl")
# println("expected duration is 30 hours")



# #fig 1 with varying variance


# τs = collect(0.1:0.05:2.5)

# pairs = [(tnormal_normal(0.0,0.2,-10.0,10.0), tnormal_normal(0.0,0.2*τ,-10.0,10.0)) for τ in τs]
# file_path = "plotscluster"
# title = "Rejection rates for 4 schemes"
# xlabel = "τ"
# ylabel = "Rej rate"
# n = 100
# m = 200
# S = 1000
# n_samples = 100
# θ = 0.05
# bootstrap = false
# file_name = "varying_variance_n=$(n)_m=$(m)_S=$(S)_permutation_n_samples=$(n_samples)"
# println("parameters are S = $S, n_samples = $(n_samples), n = $(n), m = $(m), n_threads = $(Threads.nthreads())")
# println("number of pairs of laws of RPMS: $(length(τs))")
# t = time()
# save_fig(pairs, τs, file_name, file_path, title, xlabel,ylabel, n,m,S,θ,n_samples,bootstrap)
# dur = time() - t
# println("total duration is $(dur/3600) hours")


# #println("for one pair cluster takes $(dur/3600) hours. ")