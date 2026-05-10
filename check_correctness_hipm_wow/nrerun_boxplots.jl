
include("../methods.jl")
using Plots, StatsPlots


# --- parameters ---
const N_RERUNS  = [5, 20, 50]  # n_rerun values to compare
const θ         = 0.05                   # significance level for the threshold
const N_SAMPLES = 1000                   # number of permutation samples in threshold_hipm
const S         = 100                    # number of dlip recomputations to measure variability
# ------------------


"""
    get_thresholds_distances(n, m)

Fix a single pair of hierarchical samples h_1, h_2 and study role of n_rerun for estimating FPR.

For each value in N_RERUNS:
  - one threshold is computed via the permutation approach on h_1, h_2.
  - S distances are computed on the same fixed h_1, h_2, so any variability across
    the S values comes purely from the randomness of dlip's optimization.

# Arguments:
    n::Int  :  number of probability measures in each hierarchical sample.
    m::Int  :  number of atoms per probability measure.

# Returns:
    thresholds::Vector{Float64}           :  one threshold per value in N_RERUNS,
                                             in the same order as N_RERUNS.
    distances::Dict{Int, Vector{Float64}} :  maps each n_rerun value to a vector of
                                             S distances computed on the fixed h_1, h_2.
"""
function get_thresholds_distances(n::Int, m::Int)
    q = normal_normal_B(1.0,1.5,1.5)
    h_1, h_2 = generate_hiersample(q, n, m), generate_hiersample(q, n, m)

    # joint interval covering both samples; passed explicitly to dlip so that
    # the threshold and the distance calls use the same grid.
    a = minimum((h_1.a, h_2.a))
    b = maximum((h_1.b, h_2.b))

    thresholds = Vector{Float64}(undef, length(N_RERUNS))
    distances = Dict{Int, Vector{Float64}}()

    for (i, n_rerun) in enumerate(N_RERUNS)
        @info "n_rerun = $n_rerun"
        # permutation threshold at significance level θ
        thresholds[i] = threshold_hipm(h_1, h_2, θ, N_SAMPLES, false; n_rerun = n_rerun) / sqrt(n / 2)
        # recompute dlip S times on the same fixed h_1, h_2: variability reflects
        # only the randomness of the optimization (controlled by n_rerun)
        distances[N_RERUNS[i]] = [dlip(h_1, h_2, a, b; n_rerun = n_rerun) for _ in 1:S]
    end 
    return thresholds, distances
end


"""
    plot_thresholds_distances(thresholds, distances)

Plot the HIPM threshold and the distribution of distances for each n_rerun value.
Boxplots show the variability of dlip across S recomputations on the same fixed
hierarchical samples; the scatter point is the corresponding permutation threshold.

# Arguments:
    thresholds::Vector{Float64}           :  one threshold per n_rerun value, in the
                                             same order as the sorted keys of distances.
    distances::Dict{Int, Vector{Float64}} :  maps each n_rerun value to a vector of
                                             distances computed on the fixed samples.
"""
function plot_thresholds_distances(thresholds::Vector{Float64}, distances::Dict{Int, Vector{Float64}})
    n_reruns = sort(collect(keys(distances)))
    positions = 1:length(n_reruns)

    # boxplot requires flat (x, y) vectors where x is the group label of each value
    x_box = vcat([fill(i, length(distances[nr])) for (i, nr) in enumerate(n_reruns)]...)
    y_box = vcat([distances[nr] for nr in n_reruns]...)

    fig = boxplot(x_box, y_box, label = "distances", color = :steelblue, alpha = 0.6,
                  xlabel = "n_rerun", ylabel = "value",
                  xticks = (positions, string.(n_reruns)), legend = false)
    scatter!(fig, positions, thresholds, label = "threshold",
             color = :red, markersize = 6, marker = :circle)
    return fig
end


Random.seed!(1234)
@time thresholds, distances = get_thresholds_distances(100, 100)
fig = plot_thresholds_distances(thresholds, distances)
savefig(fig, joinpath(@__DIR__, "nrerun_boxplots.png"))



