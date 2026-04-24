using CSV
using DataFrames
using Statistics
using Distributions
using Plots

function extract_label(filename::AbstractString)
    m = match(r"label=([^_\.]+)", filename)
    return m === nothing ? "unknown" : m.captures[1]
end

function mean_ci(x)
    n = length(x)
    μ = mean(x)

    if n == 1
        return (mean = μ, lower = μ, upper = μ)
    end

    s = std(x)
    se = s / sqrt(n)
    tcrit = quantile(TDist(n - 1), 0.975)

    lower = μ - tcrit * se
    upper = μ + tcrit * se

    return (mean = μ, lower = lower, upper = upper)
end

function summarize_by_m(df::DataFrame, value_col::Symbol)
    gdf = groupby(df, :m)
    out = combine(gdf, value_col => mean_ci => AsTable)
    sort!(out, :m)
    return out
end



function compute_global_ylims(folder::String)
    ymin, ymax = Inf, -Inf

    for filename in readdir(folder)
        if startswith(filename, "fp_per_m") && endswith(filename, ".csv")
            df = CSV.read(joinpath(folder, filename), DataFrame)

            if all(col -> col in names(df), ["repeat","m","fp_hipm","fp_wow"])
                hipm = summarize_by_m(df, :fp_hipm)
                wow  = summarize_by_m(df, :fp_wow)

                ymin = min(ymin,
                    minimum(hipm.lower), minimum(wow.lower)
                )
                ymax = max(ymax,
                    maximum(hipm.upper), maximum(wow.upper)
                )
            end
        end
    end

    return (ymin, ymax)
end

function make_plot_for_file(folder::String, filename::String, output_dir::String, ylims)
    base = splitext(filename)[1]
    filepath = joinpath(folder, filename)
    df = CSV.read(filepath, DataFrame)

    required_cols = ["repeat", "m", "fp_hipm", "fp_wow"]
    missing_cols = setdiff(required_cols, names(df))
    if !isempty(missing_cols)
        @warn "Skipping $filename because required columns are missing: $missing_cols"
        return
    end

    hipm_summary = summarize_by_m(df, :fp_hipm)
    wow_summary  = summarize_by_m(df, :fp_wow)

    label = extract_label(filename)
    plot_title = "fp_per_m_$(label)"

    p = plot(
        hipm_summary.m,
        hipm_summary.mean;
        ribbon = (
            hipm_summary.mean .- hipm_summary.lower,
            hipm_summary.upper .- hipm_summary.mean
        ),
        label = "HIPM",
        xlabel = "m",
        ylabel = "false positive rate",
        title = plot_title,
        linewidth = 2,
        marker = :circle,
        markersize = 4,
        ylim = ylims
    )

    plot!(
        p,
        wow_summary.m,
        wow_summary.mean;
        ribbon = (
            wow_summary.mean .- wow_summary.lower,
            wow_summary.upper .- wow_summary.mean
        ),
        label = "WOW",
        linewidth = 2,
        marker = :square,
        markersize = 4
    )

    # ------------------------------------------------------------
    # NEW: horizontal reference line at 0.05
    # ------------------------------------------------------------
    hline!(p, [0.05]; linestyle = :dash, label = "0.05")

    outfile = joinpath(output_dir, base * ".png")
    savefig(p, outfile)

    println("Saved plot: $outfile")
end

function make_plots_from_folder(; 
    input_dir = joinpath(pwd(), "values", "fp_per_m"),
    output_dir = joinpath(pwd(), "plots", "fp_per_m")
)   
    mkpath(output_dir)

    # compute shared y-limits once
    ylims = compute_global_ylims(input_dir)

    for filename in readdir(input_dir)
        if startswith(filename, "fp_per_m") && endswith(filename, ".csv")
            make_plot_for_file(input_dir, filename, output_dir, ylims)
        end
    end
end

make_plots_from_folder(    
)