# =============================================================================
# plot_pvalues.jl — plots mortality dataset p-values
# =============================================================================
#
# Reads the p-value files saved by mortality/pvalues.jl from input_dir, and plots p-value
# against year for all five methods on the same plot, together with a dashed line at
# θ = 0.05. One PNG is saved per gender to output_dir.
#
# Note: input_dir defaults to <project_root>/values/mortality_dataset, which is exactly
# where mortality/pvalues.jl saves its output — everything here runs locally, so there's
# no cluster move/copy step needed (unlike the fp/tp/mean/variance/counterexample plots).

# To use it, just run the file. It will automatically save one plot per gender
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


function plot_pvalues(gender::String;
    input_dir  = joinpath(pwd(), "values", "mortality_dataset"),
    output_dir = joinpath(pwd(), "plots", "mortality_dataset")
)
    mkpath(output_dir)

    methods  = ["hipm", "wow", "averaging", "pooling", "frechet"]
    labels   = ["HIPM", "WoW", "Averaging", "Pooling", "Fréchet"]
    colors   = [:green, :brown, :orange, :blue, :purple]
    markers  = [:square, :diamond, :dtriangle, :utriangle, :circle]

    sc = scatter(
        title    = "P-values",
        xlabel   = "Year",
        ylabel   = "p-value",
        ylims    = (-0.015, 0.25),
        legend   = :topright,
        dpi      = 300,
        size     = (800, 500)
    )

    for (method, label, color, marker) in zip(methods, labels, colors, markers)
        pvalues, time_periods = load_pvalues("pvalues_$(gender)_$(method)"; input_dir)
        scatter!(sc, time_periods, pvalues; label = label, color = color, marker = marker)
    end

    hline!(sc, [0.05]; linestyle = :dash, color = :black, label = "θ = 0.05")

    outfile = joinpath(output_dir, "pvalues_$(gender).png")
    savefig(sc, outfile)
    println("Saved plot: $outfile")
end


plot_pvalues("females")
plot_pvalues("males")