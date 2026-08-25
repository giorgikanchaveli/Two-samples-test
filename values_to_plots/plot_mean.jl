# =============================================================================
# plot_mean.jl — plots mean simulation results
# =============================================================================
#
# Reads the CSV dataframe saved by mean.jl (columns δs, hipm, wow, dm, energy)
# from input_dir — the file whose name starts with "mean" — and plots
# HIPM, WoW, DM and Energy rejection rate against δ on the same plot.
# One PNG is saved to output_dir.
#
# Note: input_dir defaults to <project_root>/values/comparison_all, not
# cluster_outputs/values where mean.jl saves its output — move/copy the
# CSV there first (see mean.jl for details).

# To use it, just run the file. It will automatically save the plot in <project_root>/plots/comparison_all.

using CSV
using DataFrames
using Plots

function plot_mean(;
    input_dir = joinpath(pwd(), "values", "comparison_all"),
    output_dir = joinpath(pwd(), "plots", "comparison_all")
)
    mkpath(output_dir)

    files = filter(f -> startswith(f, "mean") && endswith(lowercase(f), ".csv"),
                   readdir(input_dir))

    if isempty(files)
        error("No CSV file starting with 'mean' found in $input_dir")
    end

    filename = files[1]
    filepath = joinpath(input_dir, filename)

    df = CSV.read(filepath, DataFrame)

    required_cols = ["δs", "dm", "hipm", "wow", "energy"]
    missing_cols = setdiff(required_cols, names(df))
    if !isempty(missing_cols)
        error("Missing required columns in $filename: $missing_cols")
    end

    fig = plot(
        title = "Rejection rates for 4 schemes",
        xlabel = "δ",
        ylabel = "Rejection rate",
        xlims = (minimum(df.δs) - 0.05, maximum(df.δs) + 0.05),
        ylims = (-0.05, 1.05),
        legend = :right,
        lw = 2.5,
        ms = 6,
        grid = true,
        framestyle = :box,
        dpi = 300,
        size = (800, 500),
        left_margin = 10Plots.mm
    )

    plot!(
        fig, df.δs, df.dm;
        label = "Fréchet",
        color = :red,
        marker = :circle
    )

    plot!(
        fig, df.δs, df.hipm;
        label = "HIPM",
        color = :green,
        marker = :square
    )

    plot!(
        fig, df.δs, df.wow;
        label = "WoW",
        color = :brown,
        marker = :diamond
    )

    plot!(
        fig, df.δs, df.energy;
        label = "Energy",
        color = :blue,
        marker = :utriangle
    )

    outname = splitext(filename)[1] * ".png"
    outfile = joinpath(output_dir, outname)
    savefig(fig, outfile)

    println("Saved plot: $outfile")
end

plot_mean()