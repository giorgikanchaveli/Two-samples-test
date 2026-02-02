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
    
    s = ArgParseSettings(description = "Run Simulations for FP,TP,ROC")

    @add_arg_table! s begin
        "--n"
            help = "number of rows n"
            arg_type = Int
            default = 10
        "--m"
            help = "number of columns m"
            arg_type = Int
            default = 10
        "--b"
            help = "Parameter b for Beta distribution"
            arg_type = Float64
            default = 1.3
    end
    return parse_args(s)
end


# All the parameters for simulations.
struct SimConfig
    n::Int
    m::Int
    b::Float64
    S::Int
    n_samples::Int
    θs::Vector{Float64}
    bootstrap::Bool
end


function run_simulation(config::SimConfig)

    @extract config : n m b S n_samples θs bootstrap
    q_1 = DP(1.0, Beta(1, 1))
    q_2 = DP(1.0, Beta(1, b))


    @info "Starting simulation for FP,TP,ROC: n = $n, m = $m, b = $b."
    fp_hipm, fp_wow = rejection_rate_hipm_wow(q_1, q_1, n, m, S, θs, n_samples, bootstrap)
    tp_hipm, tp_wow = rejection_rate_hipm_wow(q_1, q_2, n, m, S, θs, n_samples, bootstrap)

    return DataFrame(θs = θs, fp_hipm = fp_hipm, fp_wow = fp_wow, tp_hipm = tp_hipm, tp_wow = tp_wow)
end

   



function save_results(df::DataFrame, config::SimConfig)
    
    @extract config : n m b S n_samples θs bootstrap

   
    file_id = "n=$(n)_m=$(m)_b=$(b)_S=$(S)_bootstrap=$(bootstrap)_n_samples=$(n_samples)"
        
    fp_plot = plot(title = "Type I error", aspect_ratio=:equal, xlims = (0,1), ylims = (0,1),
         xlabel = "Significance level", ylabel = "Type I error")
    plot!(fp_plot, [0, 1], [0, 1], linestyle=:dash, color=:gray,label=false)
    plot!(fp_plot, θs, df.fp_hipm, label = "HIPM")
    plot!(fp_plot, θs, df.fp_wow, label = "WoW")

    tp_plot = plot(title = "Power", aspect_ratio=:equal, xlims = (0,1), ylims = (0,1),
         xlabel = "Significance level", ylabel = "Power")
    plot!(tp_plot, [0, 1], [0, 1], linestyle=:dash, color=:gray,label=false)
    plot!(tp_plot, θs, df.tp_hipm, label = "HIPM")
    plot!(tp_plot, θs, df.tp_wow, label = "WoW")

    roc_plot = plot(title = "ROC", aspect_ratio=:equal, xlims = (0,1), ylims = (0,1),
         xlabel = "Type I error", ylabel = "Power")
    plot!(roc_plot, [0, 1], [0, 1], linestyle=:dash, color=:gray,label=false)
    plot!(roc_plot, df.fp_hipm, df.tp_hipm, label = "HIPM")
    plot!(roc_plot, df.fp_wow, df.tp_wow, label = "WoW")

    output_dir = "plotscluster"
    mkpath(output_dir) # create folder if it's missing

    savefig(fp_plot,  joinpath(output_dir, "fp_$(file_id).png"))
    savefig(tp_plot,  joinpath(output_dir, "tp_$(file_id).png"))
    savefig(roc_plot, joinpath(output_dir, "roc_$(file_id).png"))
    # csv_path  = joinpath(output_dir, file_base * ".csv") # to save also returned values
    @info "Plot saved successfully" path=output_dir
end


function main()
    parsed_args = parse_commandline()
    S = 5
    n_samples = 5
    θs = collect(0.0:0.01:1.0)
    bootstrap = false
    
    config = SimConfig(
        parsed_args["n"],
        parsed_args["m"],
        parsed_args["b"],
        S, n_samples, θs, bootstrap
        )

    df = run_simulation(config)
    save_results(df, config)

end

@info "number of threads: $(Threads.nthreads())"
t_start = time()
main()
@info "Total duration: $(round(time() - t_start, digits=2)) seconds"


