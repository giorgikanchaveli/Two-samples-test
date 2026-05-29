# Run from project root: julia simulations/power_diff.jl
#
# For each δ in a grid, computes power of HIPM and WoW at fixed θ = 0.05,
# then plots both power curves on the same axes.
#
# Q_1 is fixed; Q_2(δ) shifts the means of the latent normals by δ (scale_mix family).

using DataFrames
using Plots

include(joinpath(pwd(), "methods.jl"))

const θ_fixed   = 0.05
const θ_vec     = [θ_fixed]
const S         = 15
const n_perm    = 70
const bootstrap = false

const n = 50
const m = 100

const Q_1    = discr_law([0.5, 0.5], [Normal(0.0, 0.2), Normal(0.0, 7.0)])
const Q_2_fn = δ -> discr_law([0.5, 0.5], [Normal(δ, 0.2), Normal(δ, 7.0)])

const δs = collect(0.0:0.5:2.0)

@info "Running scale_mix power vs δ experiment (n=$n, m=$m, S=$S)"
t_start = time()

results = map(δs) do δ
    hipm, wow = rejection_rate_hipm_wow(Q_1, Q_2_fn(δ), n, m, S, θ_vec, n_perm, bootstrap)
    @info "Running experiment for δ = : $(δ)"
    return (hipm, wow)
end

df = DataFrame(
    δs   = δs,
    hipm = [r[1] for r in results],
    wow  = [r[2] for r in results],
)

@info "Done in $(round(time() - t_start, digits=2))s"
println(df)

fig = plot(
    title  = "Power at θ=$(θ_fixed) — scale_mix (n=$(n), m=$(m))",
    xlabel = "δ",
    ylabel = "Power",
    xlims  = (minimum(δs) - 0.05, maximum(δs) + 0.05),
    ylims  = (-0.05, 1.05),
    legend = :bottomright,
)
plot!(fig, df.δs, df.hipm, label = "HIPM", color = "green", linewidth = 2, marker = :circle)
plot!(fig, df.δs, df.wow,  label = "WoW",  color = "brown", linewidth = 2, marker = :square)

display(fig)