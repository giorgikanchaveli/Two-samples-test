# =============================================================================
# plot_pvalues.jl — plots mortality dataset p-values
# =============================================================================
#
# Reads the p-value files saved by mortality/pvalues.jl from input_dir, and plots p-value
# against year, together with a dashed line at θ = 0.05. Five plots are saved per gender to
# output_dir: HIPM, WoW, averaging and pooling (pvalues_<gender>_hipm_wow_av_pool.png),
# HIPM, WoW and Fréchet (pvalues_<gender>_hipm_wow_frechet.png), and Fréchet, HIPM and WoW
# each alone (pvalues_<gender>_frechet.png, pvalues_<gender>_hipm.png, pvalues_<gender>_wow.png).
#
# Note: input_dir defaults to <project_root>/values/mortality_dataset, which is exactly
# where mortality/pvalues.jl saves its output — everything here runs locally, so there's
# no cluster move/copy step needed (unlike the fp/tp/mean/variance/counterexample plots).

# To use it, just run the file. It will automatically save all five plots per gender
# (females, males) in <project_root>/plots/mortality_dataset.

using DelimitedFiles
using Plots


function load_pvalues(file_name::String;
    input_dir = joinpath(pwd(), "values", "mortality_dataset")
)
    filepath = joinpath(input_dir, "$(file_name).txt")
    data, _ = readdlm(filepath, ' ', Float64; header=true)
    return data[:, 1], Int.(data[:, 2])
end


# label, color and marker per method, so a method looks the same in every plot
const METHOD_STYLES = Dict(
    "hipm"      => ("HIPM",      :green,  :square),
    "wow"       => ("WoW",       :brown,  :diamond),
    "averaging" => ("Averaging", :orange, :dtriangle),
    "pooling"   => ("Pooling",   :blue,   :utriangle),
    "frechet"   => ("Fréchet",   :purple, :circle),
)


function plot_pvalues(gender::String, methods::Vector{String}, plot_name::String;
    ymax       = 0.3, # upper y-axis limit
    input_dir  = joinpath(pwd(), "values", "mortality_dataset"),
    output_dir = joinpath(pwd(), "plots", "mortality_dataset")
)
    mkpath(output_dir)

    sc = scatter(
        title    = "P-values",
        xlabel   = "Year",
        ylabel   = "p-value",
        ylims    = (-0.015, ymax),
        legend   = :topright,
        dpi      = 300,
        size     = (800, 500)
    )

    for method in methods
        label, color, marker = METHOD_STYLES[method]
        pvalues, time_periods = load_pvalues("pvalues_$(gender)_$(method)"; input_dir)
        scatter!(sc, time_periods, pvalues; label = label, color = color, marker = marker)
    end

    hline!(sc, [0.05]; linestyle = :dash, color = :black, label = "θ = 0.05")

    outfile = joinpath(output_dir, "pvalues_$(gender)_$(plot_name).png")
    savefig(sc, outfile)
    println("Saved plot: $outfile")
end


# Fréchet p-values go up to about 0.35, so plots with Fréchet need a higher y-axis limit
for gender in ["females", "males"]
    plot_pvalues(gender, ["hipm", "wow", "averaging", "pooling"], "hipm_wow_av_pool")
    plot_pvalues(gender, ["hipm", "wow", "frechet"], "hipm_wow_frechet"; ymax = 0.4)
    plot_pvalues(gender, ["frechet"], "frechet"; ymax = 0.4)
    plot_pvalues(gender, ["hipm"], "hipm")
    plot_pvalues(gender, ["wow"], "wow")
end