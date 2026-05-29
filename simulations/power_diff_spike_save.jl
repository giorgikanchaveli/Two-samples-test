using Plots
using DataFrames
using ArgParse
using ExtractMacro
using CSV

include(joinpath(pwd(), "methods.jl"))

function parse_commandline()
    s = ArgParseSettings(description = "Power vs spike gap for HIPM and WoW at fixed θ=0.05")
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
            help = "Number of Monte Carlo iterations S"
            arg_type = Int
            default = 1
        "--n_samples"
            help = "Number of permutation samples"
            arg_type = Int
            default = 1
    end
    return parse_args(s)
end


struct SimConfig
    n::Int
    m::Int
    S::Int
    n_samples::Int
    θ::Float64
    bootstrap::Bool
end


function run_simulation(config::SimConfig)
    gs = collect(0.0:1.0:10.0)
    θ_vec = [config.θ]
    @extract config : n m S n_samples bootstrap

    @info "Starting simulation for power_diff_spike with parameters: n=$n, m=$m, S=$S, n_samples=$n_samples."

    Q_2    = unif_spike(100.0, 0.2,  0.0)
    Q_1_fn = g -> unif_spike(100.0, 0.2, -g)

    results = map(gs) do g
        hipm, wow = rejection_rate_hipm_wow(Q_1_fn(g), Q_2, n, m, S, θ_vec, n_samples, bootstrap)
        @info "gap = $g  →  hipm = $hipm,  wow = $wow"
        return (hipm, wow)
    end

    return DataFrame(
        gap  = gs,
        hipm = [r[1] for r in results],
        wow  = [r[2] for r in results],
    )
end


function save_results(df::DataFrame, config::SimConfig)
    @extract config : n m S n_samples θ bootstrap

    values_path = joinpath(pwd(), "cluster_outputs", "values")
    plots_path  = joinpath(pwd(), "cluster_outputs", "plots")
    mkpath(values_path)
    mkpath(plots_path)

    file_name = "power_diff_spike_n=$(n)_m=$(m)_S=$(S)_bootstrap=$(bootstrap)_n_samples=$(n_samples)"

    csv_path = joinpath(values_path, file_name * ".csv")
    CSV.write(csv_path, df)
    @info "DataFrame saved successfully" path=csv_path

    fig = plot(
        title  = "Power at θ=$(θ) — unif_spike (n=$(n), m=$(m))",
        xlabel = "spike gap g",
        ylabel = "Power",
        xlims  = (minimum(df.gap) - 0.3, maximum(df.gap) + 0.3),
        ylims  = (-0.05, 1.05),
        legend = :bottomright,
    )
    plot!(fig, df.gap, df.hipm, label = "HIPM", color = "green", linewidth = 2)
    plot!(fig, df.gap, df.wow,  label = "WoW",  color = "brown",  linewidth = 2)

    png_path = joinpath(plots_path, file_name * ".png")
    savefig(fig, png_path)
    @info "Plot saved successfully" path=png_path
end


function main()
    θ         = 0.05
    bootstrap = false
    parsed    = parse_commandline()

    config = SimConfig(
        parsed["n"],
        parsed["m"],
        parsed["S"],
        parsed["n_samples"],
        θ, bootstrap,
    )

    df = run_simulation(config)
    save_results(df, config)
end


@info "Number of threads: $(Threads.nthreads())"
t_start = time()
main()
@info "Total duration: $(round(time() - t_start, digits=2)) seconds"
