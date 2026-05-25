using Plots
using DataFrames
using ArgParse
using ExtractMacro
using CSV



include(joinpath(pwd(),"methods.jl"))

function parse_commandline()

    s = ArgParseSettings(description = "Run Simulations for Rademacher section")

    @add_arg_table! s begin
        "--S"
            help = "Number of simulations S to estimate empirical threshold"
            arg_type = Int
            default = 100
        "--n_rerun"
            help = "Number of reruns for dlip optimization"
            arg_type = Int
            default = 5
    end
    return parse_args(s)
end


struct SimConfig
    S::Int
    n_rerun::Int
    θ::Float64
end



function rademacher_threshold(n::Int, m::Int, θ::Float64)::Float64
    
    a = 640 * sqrt(log(2)) / sqrt(m)
    b = 1280 * sqrt(log(2)) / sqrt(n) 
    c = 2 * (sqrt(log(2/θ) / 2) + sqrt(2 * log(4 / θ))) / sqrt(n)
    return a + b + c
end


# Given Q^1, Q^2, n, m, θ, S: simulate dlip(Q^1_{n,m}, Q^2_{n,m}) S times and return (1-θ) quantile.
function simulate_empirical_threshold(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, θ::Float64, S::Int; n_rerun::Int = 10)
    dlip_values = zeros(S)
    for s in 1:S
        h_1 = generate_hiersample(q_1, n, m)
        h_2 = generate_hiersample(q_2, n, m)
        a = min(h_1.a, h_2.a)
        b = max(h_1.b, h_2.b)
        dlip_values[s] = dlip(h_1.atoms, h_2.atoms, a, b; n_rerun)
    end
    return quantile(dlip_values, 1 - θ)
end


function run_simulation(config::SimConfig)
    ns_sim   = [collect(10:100:1000);1000]         # n values for empirical vs theoretical comparison
    ns_large = collect(10:200000:5000000)         # n values for solo theoretical decay plot

    q_1 = DP(1.0, Uniform(0.0, 1.0))
    q_2 = DP(1.0, Beta(2.0, 2.0))

    @extract config : S n_rerun θ

    @info "Starting Rademacher simulation with S = $S, n_rerun = $n_rerun, θ = $θ"

    empirical_thresholds  = zeros(length(ns_sim))
    theoretical_thresholds_sim = zeros(length(ns_sim))

    for (i, n) in enumerate(ns_sim)
        m = n  # set n = m
        empirical_thresholds[i]       = simulate_empirical_threshold(q_1, q_2, n, m, θ, S; n_rerun)
        theoretical_thresholds_sim[i] = rademacher_threshold(n, m, θ)
        @info "Progress $i / $(length(ns_sim)), n = m = $n"
    end

    theoretical_thresholds_large = [rademacher_threshold(n, n, θ) for n in ns_large]

    df_comparison  = DataFrame(
        ns          = ns_sim,
        empirical   = empirical_thresholds,
        theoretical = theoretical_thresholds_sim
    )
    df_theoretical = DataFrame(
        ns          = ns_large,
        theoretical = theoretical_thresholds_large
    )

    return df_comparison, df_theoretical
end


function save_results(df_comparison::DataFrame, df_theoretical::DataFrame, config::SimConfig)

    @extract config : S n_rerun θ

    values_path = joinpath(pwd(), "values", "rademacher_section")
    plots_path  = joinpath(pwd(), "plots", "rademacher_section")
    mkpath(values_path)
    mkpath(plots_path)

    file_name = "rademacher_emp_vs_theor_S=$(S)"

    CSV.write(joinpath(values_path, file_name * ".csv"), df_comparison)
    @info "Comparison dataframe saved" path=joinpath(values_path, file_name * ".csv")

    CSV.write(joinpath(values_path, "rademacher_thresholds.csv"), df_theoretical)
    @info "Theoretical dataframe saved" path=joinpath(values_path, "rademacher_thresholds.csv")

    # Shared y range across both plots (both on log scale)
    log_emp   = log.(df_comparison.empirical)
    log_theor_sim   = log.(df_comparison.theoretical)
    log_theor = log.(df_theoretical.theoretical)
    y_all = vcat(log_emp, log_theor_sim, log_theor)
    y_top = maximum(y_all)
    y_bot = minimum(y_all)
    shared_ylims = (min(y_bot, 0.0) - 0.05 * y_top, y_top * 1.05)

    # Plot 1: empirical and theoretical thresholds per n (= m)
    fig1 = plot(
        title  = "Empirical vs Theoretical Rademacher Thresholds",
        xlabel = "n = m",
        ylabel = "log(Threshold)",
        xlims  = (minimum(df_comparison.ns) - 50, maximum(df_comparison.ns) + 50),
        ylims  = shared_ylims
    )
    plot!(fig1, df_comparison.ns, log_emp,         label = "Empirical",   color = "green", marker = (:circle, 4))
    plot!(fig1, df_comparison.ns, log_theor_sim,   label = "Theoretical", color = "red",   marker = (:circle, 4))
    fig1_path = joinpath(plots_path, file_name * "_comparison.png")
    savefig(fig1, fig1_path)
    @info "Comparison plot saved" path=fig1_path

    # Plot 2: solo theoretical threshold decay over large n grid, y-axis in logarithms
    fig2 = plot(
        title  = "Rademacher Thresholds",
        xlabel = "n = m",
        ylabel = "log(Threshold)",
        xlims  = (-0.03 * maximum(df_theoretical.ns), maximum(df_theoretical.ns) + 5e5),
        ylims  = shared_ylims
    )
    plot!(fig2, df_theoretical.ns, log_theor, label = "hipm", color = "blue", marker = (:circle, 4))
    fig2_path = joinpath(plots_path, "theoretical_thresholds.png")
    savefig(fig2, fig2_path)
    @info "Log-scale plot saved" path=fig2_path

    # Combined: both plots side by side
    fig_combined = plot(fig1, fig2, layout = (1, 2), size = (1400, 500))
    combined_path = joinpath(plots_path,  "Rademacher_combined.png")
    savefig(fig_combined, combined_path)
    @info "Combined plot saved" path=combined_path
end


function main()
    θ = 0.05
    parsed_args = parse_commandline()

    config = SimConfig(
        parsed_args["S"],
        parsed_args["n_rerun"],
        θ
    )

    df_comparison, df_theoretical = run_simulation(config)
    save_results(df_comparison, df_theoretical, config)
end

@info "number of threads: $(Threads.nthreads())"
t_start = time()
main()
@info "Total duration: $(round(time() - t_start, digits=2)) seconds"
