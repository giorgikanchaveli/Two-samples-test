using CSV
using DataFrames
using Plots

function extract_label(filename::AbstractString)
    m = match(r"label=([^_\.]+)", filename)
    return m === nothing ? "unknown" : m.captures[1]
end

function make_plot_for_file(folder::String, filename::String, outdir::String)
    base = splitext(filename)[1]
    filepath = joinpath(folder, filename)
    df = CSV.read(filepath, DataFrame)
    required_cols = ["ms", "fp_hipm", "fp_wow"]
    missing_cols = setdiff(required_cols, names(df))
    if !isempty(missing_cols)
        @warn "Skipping $filename because required columns are missing: $missing_cols"
        return
    end

    label = extract_label(filename)
    plot_title = "fp_per_m_$(label)"

    p = scatter(
        df.ms,
        df.fp_hipm;
        label = "fp_hipm",
        xlabel = "ms",
        ylabel = "value",
        title = plot_title,
        markersize = 4
    )

    scatter!(
        p,
        df.ms,
        df.fp_wow;
        label = "fp_wow",
        markersize = 4
    )

    outfile = joinpath(outdir, base*".png")
    savefig(p, outfile)

    println("Saved plot: $outfile")
end

function make_plots_from_folder(folder::String, outdir::String)
    mkpath(outdir)
    for filename in readdir(folder)
        if startswith(filename, "fp_per_m")
            make_plot_for_file(folder, filename, outdir)
        end
    end
end

make_plots_from_folder(joinpath(pwd(), "values", "permutation_simulations"),
                        joinpath(pwd(), "plots", "permutation_simulations", "fp_per_m"))