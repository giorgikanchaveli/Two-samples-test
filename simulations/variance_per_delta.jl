using Distributions
using Plots
using Statistics

include(joinpath(pwd(), "methods.jl"))



# Q_1 = discr_law([0.5, 0.5], [Normal(0.0, 0.2), Normal(0.0, 7.0)])
# Q_2 = discr_law([0.5, 0.5], [Normal(1.3, 0.2), Normal(1.3, 7.0)])

# L = 50.0
# Δ = 1.0
# ϵ = 0.1


# Q_1 = discr_law([0.5, 0.5], [Normal(0.0, ϵ), Normal(0.0, L)])
# Q_2 = discr_law([0.5, 0.5], [Normal(Δ, ϵ), Normal(0.0, L)])


m  = 250
n = 50
S  = 20
δs = collect(0.0:0.1:1.5)

@info "variance_per_δ"
t_start = time()

vars_wow  = zeros(length(δs))
vars_hipm = zeros(length(δs))

for (k, δ) in enumerate(δs)
    wow_vals  = zeros(S)
    hipm_vals = zeros(S)
    Q_1 = discr_law([0.5, 0.5], [Normal(0.0, 0.2), Normal(0.0, 5.0)])
    Q_2 = discr_law([0.5, 0.5], [Normal(δ, 0.2), Normal(δ, 5.0)])

    @floop ThreadedEx() for s in 1:S
        h_1 = generate_hiersample(Q_1, n, m)
        h_2 = generate_hiersample(Q_2, n, m)
        a   = minimum((h_1.a, h_2.a))
        b   = maximum((h_1.b, h_2.b))
        wow_vals[s]  = ww(h_1, h_2)
        hipm_vals[s] = dlip(h_1, h_2, a, b)
    end

    vars_wow[k]  = var(wow_vals)
    vars_hipm[k] = var(hipm_vals)
    @info "n=$n  var(WoW)=$(round(vars_wow[k], digits=5))  var(HIPM)=$(round(vars_hipm[k], digits=5))"
end

@info "Total time: $(round(time() - t_start, digits=2))s"

fig = plot(
    title   = "Variance of WoW and HIPM vs δ ",
    xlabel  = "δ",
    ylabel  = "Variance of distance",
    xlims   = (minimum(δs) - 0.05, maximum(δs) + 0.05),
    legend  = :topright,
)
plot!(fig, δs, vars_wow,  label = "WoW",  color = "brown", linewidth = 2, marker = :square)
plot!(fig, δs, vars_hipm, label = "HIPM", color = "green", linewidth = 2, marker = :circle)

display(fig)
