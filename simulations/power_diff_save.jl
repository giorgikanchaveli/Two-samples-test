using Plots
using DataFrames
using ArgParse
using ExtractMacro
using CSV

include(joinpath(pwd(), "methods.jl"))

function parse_commandline()
    s = ArgParseSettings(description = "Power vs δ for HIPM and WoW at fixed θ=0.05")
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
    δs = collect(0.0:0.2:2.0)
    θ_vec = [config.θ]
    @extract config : n m S n_samples bootstrap

    @info "Starting simulation for power_diff with parameters: n=$n, m=$m, S=$S, n_samples=$n_samples."

    Q_1    = discr_law([0.5, 0.5], [Normal(0.0, 0.2), Normal(0.0, 7.0)])
    Q_2_fn = δ -> discr_law([0.5, 0.5], [Normal(δ, 0.2), Normal(δ, 7.0)])

    results = map(δs) do δ
        hipm, wow = rejection_rate_hipm_wow(Q_1, Q_2_fn(δ), n, m, S, θ_vec, n_samples, bootstrap)
        return (hipm, wow)
    end

    return DataFrame(
        δs   = δs,
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

    file_name = "power_diff_n=$(n)_m=$(m)_S=$(S)_bootstrap=$(bootstrap)_n_samples=$(n_samples)"

    values_path = joinpath(values_path, file_name * ".csv")
    CSV.write(values_path, df)
    @info "DataFrame saved successfully" path=values_path

    fig = plot(
        title  = "Power at θ=$(θ) — scale_mix (n=$(n), m=$(m))",
        xlabel = "δ",
        ylabel = "Power",
        xlims  = (minimum(df.δs) - 0.05, maximum(df.δs) + 0.05),
        ylims  = (-0.05, 1.05),
        legend = :bottomright,
    )
    plot!(fig, df.δs, df.hipm, label = "HIPM", color = "green", linewidth = 2)
    plot!(fig, df.δs, df.wow,  label = "WoW",  color = "brown",  linewidth = 2)

    plots_path = joinpath(plots_path, file_name * ".png")
    savefig(fig, plots_path)
    @info "Plot saved successfully" path=plots_path
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
