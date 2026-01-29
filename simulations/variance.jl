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
    
    s = ArgParseSettings(description = "Run Simulations for variance")

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
