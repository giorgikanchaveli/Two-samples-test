# Run from project root: julia simulations/power_diff_spike.jl
#
# For each spike gap g in a grid, computes power of HIPM and WoW at fixed θ = 0.05,
# then plots both power curves on the same axes.
#
# Q_1: ν = Unif[10,110], spike at -g  →  P_u^1 = 0.8δ_u + 0.2δ_{-g}
# Q_2: ν = Unif[10,110], spike at  0  →  P_u^2 = 0.8δ_u + 0.2δ_0
#
# At gap g:  W₁(row₁, row₂) = 0.2g + 0.8|U_i - V_j|
#   WoW signal = 0.2g, WoW noise std ≈ 12.6  →  SNR_WoW ≈ 0.016g
#   HIPM uses f*(x) = min(x+g, g), flat on [10,110]  →  SNR_HIPM ≈ 1.59g

using DataFrames
using Plots

include(joinpath(pwd(), "methods.jl"))

const θ_fixed   = 0.05
const θ_vec     = [θ_fixed]
const S         = 15
const n_perm    = 70
const bootstrap = false

const n = 30
const m = 50

const Q_2    = unif_spike(50.0, 0.2,  0.0)
const Q_1_fn = g -> unif_spike(50.0, 0.2, -g)

const gs = collect(0.0:2.0:10.0)   # spike gap: 0, 2, 4, 6, 8, 10

@info "Running unif_spike power vs gap experiment (n=$n, m=$m, S=$S)"
t_start = time()

results = map(gs) do g
    hipm, wow = rejection_rate_hipm_wow(Q_1_fn(g), Q_2, n, m, S, θ_vec, n_perm, bootstrap)
    @info "gap = $g  →  hipm = $hipm,  wow = $wow"
    return (hipm, wow)
end

df = DataFrame(
    gap  = gs,
    hipm = [r[1] for r in results],
    wow  = [r[2] for r in results],
)

@info "Done in $(round(time() - t_start, digits=2))s"
println(df)

fig = plot(
    title  = "Power at θ=$(θ_fixed) — unif_spike (n=$(n), m=$(m))",
    xlabel = "spike gap g",
    ylabel = "Power",
    xlims  = (minimum(gs) - 0.3, maximum(gs) + 0.3),
    ylims  = (-0.05, 1.05),
    legend = :bottomright,
)
plot!(fig, df.gap, df.hipm, label = "HIPM", color = "green", linewidth = 2, marker = :circle)
plot!(fig, df.gap, df.wow,  label = "WoW",  color = "brown",  linewidth = 2, marker = :square)

display(fig)
