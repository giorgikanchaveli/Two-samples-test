# =============================================================================
# plot_all.jl — runs every plotting script in values_to_plots/
# =============================================================================
#
# Instead of running each plot_*.jl file by hand, run this one: it runs all of them,
# so every plot is saved into its usual folder in <project_root>/plots/.
#
# Each script runs in its own module, so functions with the same name in different
# scripts don't clash, and running this file twice in the same REPL works.
# If one script fails (e.g. its CSV is not in values/ yet), the others still run, and
# the failed ones are listed at the end.
#
# HOW TO RUN (from anywhere):
#   julia --project=. values_to_plots/plot_all.jl
# or from VS Code / the Julia REPL: include("values_to_plots/plot_all.jl")

const PLOT_SCRIPTS = [
    "plot_tpfp_csv.jl",
    "plot_mean.jl",
    "plot_variance.jl",
    "plot_counterexample.jl",
    "plot_rademacher.jl",
    "plot_pvalues.jl",
]

failed = String[]
cd(dirname(@__DIR__)) do # the scripts build their paths from pwd(), so run them from the project root
    for script in PLOT_SCRIPTS
        println("\n=== $script")
        try
            Base.include(Module(), joinpath(@__DIR__, script))
        catch err
            println("FAILED: ", sprint(showerror, err))
            push!(failed, script)
        end
    end
end

println()
if isempty(failed)
    println("All $(length(PLOT_SCRIPTS)) plotting scripts ran.")
else
    println("$(length(failed)) of $(length(PLOT_SCRIPTS)) plotting scripts failed: ", join(failed, ", "))
end
