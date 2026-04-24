using CSV, DataFrames, Plots

# df = CSV.read("values/permutation_simulations/tp_label_1=3_label_2=6_n=100_m=100_S=1000_n_perm=100.csv", DataFrame)
# p = plot(
#         df[!, "θs"],
#         df[!, "tp_hipm"],
#         label = "hipm",
#         linewidth = 2,
#         marker = :circle,
#         xlabel = "θ",
#         ylabel = "True Positive Rate",
#         title = "tp_3b",
#         legend = :best,
#         aspect_ratio = :equal,
#         xlims = (0, 1),
#         ylims = (0, 1)
#     )
# plot!(
#         p,
#         df[!, "θs"],
#         df[!, "tp_wow"],
#         label = "wow",
#         linewidth = 2,
#         marker = :square
#     )

# savefig(p, "tp_label_3b_n=100_m=100_S=1000_n_perm=100.png")

# read data
df = CSV.read("values/counterexample_n=100_m=200_S=1000_bootstrap=false_n_samples=100.csv", DataFrame)

fig = plot(
    title = "Rejection rates for 3 schemes",
    xlabel = "λ",
    ylabel = "Rej rate",
    xlims = (minimum(df.λs) - 0.05, maximum(df.λs) + 0.05),
    ylims = (-0.05, 1.05),
    legend = :right,
    lw = 2.5,
    ms = 6,
    grid = true,
    framestyle = :box,
    dpi = 300,
    size = (800, 500),
    left_margin = 10Plots.mm
)

plot!(
    fig, df.λs, df.dm,
    label = "DM",
    color = :red,
    marker = :circle
)

# plot!(
#     fig, df.τs, df.energy,
#     label = "Energy",
#     color = :blue,
#     marker = :utriangle
# )

plot!(
    fig, df.λs, df.hipm,
    label = "HIPM",
    color = :green,
    marker = :square
)

plot!(
    fig, df.λs, df.wow,
    label = "WoW",
    color = :brown,
    marker = :diamond
)


savefig(fig, "counterexample_n=100_m=200_S=1000_bootstrap=false_n_samples=100.png")
