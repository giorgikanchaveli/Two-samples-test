# Run from project root: julia simulations/local_experiment.jl

using DataFrames
using Plots

include(joinpath(pwd(), "methods.jl"))

# ==============================================================================
# CONFIGURATION — edit this section to define your experiments
# ==============================================================================

S      = 20
n_perm = 70

# False positive experiments: Q vs Q (same distribution on both sides)
fp_experiments = [
    # (label = "DP_Unif", Q = DP(1.0, Uniform(0, 1)),    n = 50, m = 5),
    # (label = "Normal",  Q = normal_normal_A(0.0, 1.0), n = 50, m = 100),
]

# ν = Unif[10,110], P_u^1 = 0.8δ_u + 0.2δ_0, P_u^2 = 0.8δ_u + 0.2δ_1
# tp_experiments = [
#     (label = "unif_spike", Q_1 = unif_spike(500.0, 0.2, 0.0),
#                            Q_2 = unif_spike(500.0, 0.2, 2.0),
#                            n = 40, m = 200),
# ]



tp_experiments = [
    # (label = "scale_mix",
    #  Q_1 = discr_law([0.5, 0.5], [Normal(0.0, 0.5), Normal(0.0, 5.0)]),
    #  Q_2 = discr_law([0.5, 0.5], [Normal(0.3, 0.5), Normal(0.3, 5.0)]),
    #  n = 15, m = 50),
    # ν = Unif[10,110]; Q1 spike at -10, Q2 spike at 0 (gap = 10).
    # WoW: every cross-population pair costs 2 + 0.8|U_i - V_j|.
    #   The 0.8|U-V| term has std ≈ 12.6 vs signal 2 → SNR_WoW ≈ 0.16.
    # HIPM: witness f*(x)=min(x+10,10) is constant on [10,110],
    #   so between-cluster variance = 0 → SNR_HIPM ≈ 15.9.
    # Grid step ≈ 0.48 on [-10,110] → 21 grid points cover the signal region,
    #   so the optimizer reliably finds the witness.
    (label = "unif_spike_hipm_win",
     Q_1 = unif_spike(100.0, 0.2, -10.0),
     Q_2 = unif_spike(100.0, 0.2,   0.0),
     n = 40, m = 50),
    (label = "scale_mix",
     Q_1 = discr_law([0.5, 0.5], [Normal(0.0, 0.2), Normal(0.0, 7.0)]),
     Q_2 = discr_law([0.5, 0.5], [Normal(1.3, 0.2), Normal(1.3, 7.0)]),
     n = 50, m = 100)
    # (label = "normal_shift",
    #  Q_1 = normal_normal_A(0.0, 1.0),
    #  Q_2 = normal_normal_A(0.5, 1.0),
    #  n = 15, m = 100),
    # (label = "normal_shift",
    #  Q_1 = normal_normal_A(0.0, 1.0),
    #  Q_2 = normal_normal_A(0.5, 1.0),
    #  n = 50, m = 100),

    # (label = "DP_shift",
    #  Q_1 = DP(2.0, Uniform(0.0, 1.0)),
    #  Q_2 = DP(2.0, Uniform(0.3, 1.3)),
    #  n = 15, m = 20),
    # (label = "DP_shift",
    #  Q_1 = DP(2.0, Uniform(0.0, 1.0)),
    #  Q_2 = DP(2.0, Uniform(0.3, 1.3)),
    #  n = 40, m = 20),

    # (label = "discrete_2nd_order",
    #  Q_1 = discr_law([1.0], [DiscreteNonParametric([-1.0, 1.0], [0.5, 0.5])]),
    #  Q_2 = discr_law([0.5, 0.5], [DiscreteNonParametric([-1.0], [1.0]),
    #                                DiscreteNonParametric([ 1.0], [1.0])]),
    #  n = 15, m = 10),
    # (label = "discrete_2nd_order",
    #  Q_1 = discr_law([1.0], [DiscreteNonParametric([-1.0, 1.0], [0.5, 0.5])]),
    #  Q_2 = discr_law([0.5, 0.5], [DiscreteNonParametric([-1.0], [1.0]),
    #                                DiscreteNonParametric([ 1.0], [1.0])]),
    #  n = 40, m = 10),
]








# ==============================================================================

const θs = collect(0.0:0.01:1.0)

function run_fp(config)
    fp_hipm, fp_wow = rejection_rate_hipm_wow(config.Q, config.Q, config.n, config.m, S, θs, n_perm, false)
    return DataFrame(θs = θs, fp_hipm = fp_hipm, fp_wow = fp_wow)
end

function run_tp(config)
    tp_hipm, tp_wow = rejection_rate_hipm_wow(config.Q_1, config.Q_2, config.n, config.m, S, θs, n_perm, false)
    return DataFrame(θs = θs, tp_hipm = tp_hipm, tp_wow = tp_wow)
end

function build_plot(df, prefix, title)
    ylabel = prefix == "tp" ? "True Positive Rate" : "False Positive Rate"
    p = plot(
        df[!, "θs"], df[!, "$(prefix)_hipm"],
        label = "hipm", linewidth = 2, marker = :circle,
        xlabel = "θ", ylabel = ylabel, title = title,
        legend = :best, aspect_ratio = :equal, xlims = (0, 1), ylims = (0, 1)
    )
    plot!(p, df[!, "θs"], df[!, "$(prefix)_wow"], label = "wow", linewidth = 2, marker = :square)
    return p
end

fp_plots = map(fp_experiments) do config
    @info "Running FP experiment: $(config.label)"
    t = @elapsed df = run_fp(config)
    @info "Done in $(round(t, digits=2))s"
    build_plot(df, "fp", "fp_$(config.label)_n=$(config.n)_m=$(config.m)")
end

tp_plots = map(tp_experiments) do config
    @info "Running TP experiment: $(config.label)"
    t = @elapsed df = run_tp(config)
    @info "Done in $(round(t, digits=2))s"
    build_plot(df, "tp", "tp_$(config.label)_n=$(config.n)_m=$(config.m)")
end

# display(fp_plots[1])






# # True positive experiments: Q_1 vs Q_2 (different distributions)
# #
# # Four experiment types, each at n=15 and n=40, to compare HIPM and WW across
# # different settings and sample size regimes.
# #
# # Expected behaviour:
# #   scale_mix    → HIPM >> WW  (scale noise drowns WW's cost matrix)
# #   normal_shift → HIPM ≈ WW  (clean signal, both methods see the same thing)
# #   DP_shift     → HIPM ≈ WW  (moderate case, no scale noise)
# #   discrete_2nd → HIPM ≈ WW  (second-order law difference, both detect it)

# tp_experiments = [

#     # ── 1. Scale-mixing: HIPM advantage ───────────────────────────────────────
#     # Each measure is narrow N(0, 0.5²) or wide N(0, 5²) with prob 1/2.
#     # Random scale is orthogonal to the mean-shift signal δ=0.3.
#     # W₁(narrow, wide) ≈ 3.59 >> std(∫x dμ) ≈ √(m/π) × scale/m.
#     # σ_WW/σ_DLIP ≈ √(m/π) × (σ_l-σ_s)/√(σ_s²+σ_l²) ≈ 4.5 for m=50.
#     (label = "scale_mix",
#      Q_1 = discr_law([0.5, 0.5], [Normal(0.0, 0.5), Normal(0.0, 5.0)]),
#      Q_2 = discr_law([0.5, 0.5], [Normal(0.3, 0.5), Normal(0.3, 5.0)]),
#      n = 15, m = 50),
#     (label = "scale_mix",
#      Q_1 = discr_law([0.5, 0.5], [Normal(0.0, 0.5), Normal(0.0, 5.0)]),
#      Q_2 = discr_law([0.5, 0.5], [Normal(0.3, 0.5), Normal(0.3, 5.0)]),
#      n = 40, m = 50),

#     # ── 2. Normal hierarchical: both comparable ────────────────────────────────
#     # θ ~ N(μ, 1), measure = N(θ, 1). Mean shift δ=0.5 in the outer distribution.
#     # No irrelevant geometric noise: W₁(N(θ₁,1), N(θ₂,1)) = |θ₁-θ₂| = ∫x dμ difference.
#     # HIPM and WW capture exactly the same signal, so power curves should overlap.
#     (label = "normal_shift",
#      Q_1 = normal_normal_A(0.0, 1.0),
#      Q_2 = normal_normal_A(0.5, 1.0),
#      n = 15, m = 20),
#     (label = "normal_shift",
#      Q_1 = normal_normal_A(0.0, 1.0),
#      Q_2 = normal_normal_A(0.5, 1.0),
#      n = 40, m = 20),

#     # ── 3. DP base measure shift: moderate case ────────────────────────────────
#     # Both DPs have concentration α=2 but shifted Uniform base measure (δ=0.3).
#     # Each DP draw has ~α·log(1+m/α) ≈ 5 distinct atoms from U(0,1) or U(0.3,1.3).
#     # Moderate W₁ variability from random atom positions; both methods face similar
#     # difficulty. Useful to compare with scale_mix at same n.
#     (label = "DP_shift",
#      Q_1 = DP(2.0, Uniform(0.0, 1.0)),
#      Q_2 = DP(2.0, Uniform(0.3, 1.3)),
#      n = 15, m = 20),
#     (label = "DP_shift",
#      Q_1 = DP(2.0, Uniform(0.0, 1.0)),
#      Q_2 = DP(2.0, Uniform(0.3, 1.3)),
#      n = 40, m = 20),

#     # ── 4. Discrete second-order difference ────────────────────────────────────
#     # Q_1: each measure is always the uniform mix on {-1,+1} (fixed shape).
#     # Q_2: each measure is either δ_{-1} or δ_{+1} (random point mass).
#     # Both have the same marginal distribution of atoms (mean=0, support={-1,+1})
#     # but differ in the variance of the law of measures (second-order difference).
#     # For large m: HIPM sees std(∫x dμ|Q₁)→0 vs std(∫x dμ|Q₂)=1; WW sees
#     # W₁(mixed, point_mass)=1 vs W₁(mixed, mixed)=0. Both detect the difference.
#     (label = "discrete_2nd_order",
#      Q_1 = discr_law([1.0], [DiscreteNonParametric([-1.0, 1.0], [0.5, 0.5])]),
#      Q_2 = discr_law([0.5, 0.5], [DiscreteNonParametric([-1.0], [1.0]),
#                                    DiscreteNonParametric([ 1.0], [1.0])]),
#      n = 15, m = 10),
#     (label = "discrete_2nd_order",
#      Q_1 = discr_law([1.0], [DiscreteNonParametric([-1.0, 1.0], [0.5, 0.5])]),
#      Q_2 = discr_law([0.5, 0.5], [DiscreteNonParametric([-1.0], [1.0]),
#                                    DiscreteNonParametric([ 1.0], [1.0])]),
#      n = 40, m = 10),
# ]
