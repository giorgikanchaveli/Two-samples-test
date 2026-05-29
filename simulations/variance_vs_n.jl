# Run from project root: julia simulations/variance_vs_n.jl
#
# Each P_i is drawn as either a sharp signal component or a wide noise component
# with equal probability:
#
#   Q_1 = discr_law([0.5, 0.5], [Normal(0, ε),  Normal(0, L)])
#   Q_2 = discr_law([0.5, 0.5], [Normal(Δ, ε),  Normal(0, L)])
#
# The noise component is identical in both laws (mean 0, std L).
# HIPM projects each P_i to its mean via f=identity, so noise rows cancel exactly
# and variance is driven by ε²/(m·n). WoW computes pairwise W_1 distances which
# fluctuate at scale L/√m for every noise-noise matched pair, giving variance 2L²/(m·n).
# Predicted ratio: Var[WoW]/Var[HIPM] ≈ 2L²/ε².

using Distributions
using Plots
using Statistics

include(joinpath(pwd(), "methods.jl"))



# Q_1 = discr_law([0.5, 0.5], [Normal(0.0, 0.2), Normal(0.0, 7.0)])
# Q_2 = discr_law([0.5, 0.5], [Normal(1.3, 0.2), Normal(1.3, 7.0)])

L = 50.0
Δ = 1.0
ϵ = 0.1


Q_1 = discr_law([0.5, 0.5], [Normal(0.0, ϵ), Normal(0.0, L)])
Q_2 = discr_law([0.5, 0.5], [Normal(Δ, ϵ), Normal(0.0, L)])


m  = 50
ns = [5, 10, 20, 30]
S  = 50

@info "variance_vs_n"
t_start = time()

vars_wow  = zeros(length(ns))
vars_hipm = zeros(length(ns))

for (k, n) in enumerate(ns)
    wow_vals  = zeros(S)
    hipm_vals = zeros(S)

    @floop ThreadedEx() for s in 1:S
        h_1 = generate_hiersample(Q_1, n, m)
        h_2 = generate_hiersample(Q_2, n, m)
        a   = minimum((h_1.a, h_2.a))
        b   = maximum((h_1.b, h_2.b))
        wow_vals[s]  = ww(h_1, h_2)
        hipm_vals[s] = dlip(h_1, h_2, a, b; n_rerun = 10, n_grid = 5000)
    end

    vars_wow[k]  = var(wow_vals)
    vars_hipm[k] = var(hipm_vals)
    @info "n=$n  var(WoW)=$(round(vars_wow[k], digits=5))  var(HIPM)=$(round(vars_hipm[k], digits=5))"
end

@info "Total time: $(round(time() - t_start, digits=2))s"

fig = plot(
    title   = "Variance of WoW and HIPM vs n ",
    xlabel  = "n",
    ylabel  = "Variance of distance",
    xlims   = (minimum(ns) - 2, maximum(ns) + 2),
    legend  = :topright,
)
plot!(fig, ns, vars_wow,  label = "WoW",  color = "brown", linewidth = 2, marker = :square)
plot!(fig, ns, vars_hipm, label = "HIPM", color = "green", linewidth = 2, marker = :circle)

display(fig)
