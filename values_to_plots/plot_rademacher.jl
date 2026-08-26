# =============================================================================
# plot_rademacher.jl — plots section_rademacher simulation results
# =============================================================================
#
# Reads two CSVs saved by section_rademacher.jl from input_dir:
#   - the file starting with "rademacher_emp_vs_theor" (columns ns, empirical,
#     theoretical) — empirical vs theoretical threshold over a moderate n = m grid.
#   - "rademacher_thresholds.csv" (columns ns, theoretical) — theoretical threshold
#     alone over a much larger n = m grid.
# It saves three PNGs to output_dir: the empirical-vs-theoretical comparison plot,
# the solo theoretical decay plot, and both combined side by side. All three share
# the same y-axis limits (log scale) for easy comparison.
#
# Note: input_dir defaults to <project_root>/values/rademacher_section, which is
# where section_rademacher.jl saves its output directly — there's no cluster run
# for this script, so no move/copy step is needed.

# To use it, just run the file. It will automatically save the plots in <project_root>/plots/rademacher_section.

using CSV
using DataFrames
using Plots

function plot_rademacher(;
    input_dir  = joinpath(pwd(), "values", "rademacher_section"),
    output_dir = joinpath(pwd(), "plots", "rademacher_section")
)
    mkpath(output_dir)

    files = filter(f -> startswith(f, "rademacher_emp_vs_theor") && endswith(lowercase(f), ".csv"),
                   readdir(input_dir))
    if isempty(files)
        error("No CSV file starting with 'rademacher_emp_vs_theor' found in $input_dir")
    end
    comparison_filename = files[1]

    df_comparison  = CSV.read(joinpath(input_dir, comparison_filename), DataFrame)
    df_theoretical = CSV.read(joinpath(input_dir, "rademacher_thresholds.csv"), DataFrame)

    # Shared y range across both plots (both on log scale)
    log_emp       = log.(df_comparison.empirical)
    log_theor_sim = log.(df_comparison.theoretical)
    log_theor     = log.(df_theoretical.theoretical)
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
    plot!(fig1, df_comparison.ns, log_emp,       label = "Empirical",   color = "green", marker = (:circle, 4))
    plot!(fig1, df_comparison.ns, log_theor_sim, label = "Theoretical", color = "red",   marker = (:circle, 4))
    fig1_path = joinpath(output_dir, splitext(comparison_filename)[1] * "_comparison.png")
    savefig(fig1, fig1_path)
    println("Saved plot: $fig1_path")

    # Plot 2: solo theoretical threshold decay over large n grid, y-axis in logarithms
    fig2 = plot(
        title  = "Rademacher Thresholds",
        xlabel = "n = m",
        ylabel = "log(Threshold)",
        xlims  = (-0.03 * maximum(df_theoretical.ns), maximum(df_theoretical.ns) + 5e5),
        ylims  = shared_ylims
    )
    plot!(fig2, df_theoretical.ns, log_theor, label = "hipm", color = "blue", marker = (:circle, 4))
    fig2_path = joinpath(output_dir, "theoretical_thresholds.png")
    savefig(fig2, fig2_path)
    println("Saved plot: $fig2_path")

    # Combined: both plots side by side
    fig_combined = plot(fig1, fig2, layout = (1, 2), size = (1400, 500))
    combined_path = joinpath(output_dir, "Rademacher_combined.png")
    savefig(fig_combined, combined_path)
    println("Saved plot: $combined_path")
end

plot_rademacher()