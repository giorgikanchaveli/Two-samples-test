using CSV
using DataFrames
using Plots

function plot_counterexample(;
    input_dir = joinpath(pwd(), "values", "comparison_with_dm"),
    output_dir = joinpath(pwd(), "plots", "comparison_with_dm")
)
    mkpath(output_dir)

    files = filter(f -> startswith(f, "counterexample") && endswith(lowercase(f), ".csv"),
                   readdir(input_dir))

    if isempty(files)
        error("No CSV file starting with 'counterexample' found in $input_dir")
    end

    filename = files[1]
    filepath = joinpath(input_dir, filename)

    df = CSV.read(filepath, DataFrame)

    required_cols = ["λs", "dm", "hipm", "wow"]
    missing_cols = setdiff(required_cols, names(df))
    if !isempty(missing_cols)
        error("Missing required columns in $filename: $missing_cols")
    end

    fig = plot(
        title = "Rejection rates for 3 schemes",
        xlabel = "λ",
        ylabel = "Rej rate",
        xlims = (minimum(df.λs) - 0.05, maximum(df.λs) + 0.05),
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
        fig, df.λs, df.dm;
        label = "DM",
        color = :red,
        marker = :circle
    )

    plot!(
        fig, df.λs, df.hipm;
        label = "HIPM",
        color = :green,
        marker = :square
    )

    plot!(
        fig, df.λs, df.wow;
        label = "WoW",
        color = :brown,
        marker = :diamond
    )

    outname = splitext(filename)[1] * ".png"
    outfile = joinpath(output_dir, outname)
    savefig(fig, outfile)

    println("Saved plot: $outfile")
end

plot_counterexample()