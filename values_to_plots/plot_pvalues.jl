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

    methods = ["hipm", "wow", "averaging", "pooling"]
    colors  = [:blue, :red, :green, :orange]

    sc = scatter(
        title    = "P-values, $(gender)",
        xlabel   = "Year",
        ylabel   = "p-value",
        ylims    = (-0.015, 0.25),
        legend   = :topright,
        dpi      = 300,
        size     = (800, 500)
    )

    for (method, color) in zip(methods, colors)
        pvalues, time_periods = load_pvalues("pvalues_$(gender)_$(method)"; input_dir)
        scatter!(sc, time_periods, pvalues; label = method, color = color)
    end

    hline!(sc, [0.05]; linestyle = :dash, color = :black, label = "θ = 0.05")

    outfile = joinpath(output_dir, "pvalues_$(gender).png")
    savefig(sc, outfile)
    println("Saved plot: $outfile")
end


plot_pvalues("females")
plot_pvalues("males")