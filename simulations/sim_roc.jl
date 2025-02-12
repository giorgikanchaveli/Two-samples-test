include("methods_roc.jl")




Random.seed!(1234)
distances = collect(0.01:0.2:0.71)
n_top = 75
n_bottom = 2
nPerms = 75
nReps = 75
d_found,tp = tp_per_d(distances, n_top, n_bottom, nPerms, nReps)
fp = fp_rate(n_top, n_bottom, nPerms, nReps)


# plot of TP only for ww
pl_ww = plot(title = "TP for WW", xlabel = "probability", ylabel = "TP rate",
         ratio = 1.0, xlims = (0,1), ylims = (0,1))
θs = collect(0.0:0.01:1)
plot!(pl_ww, θs, θs)
for i in 1:length(d_found)
    plot!(pl_ww, θs, tp[i][1], label = "d = $(d_found[i])")
end

# plot of TP only for lip
pl_lip = plot(title = "TP for HIPM", xlabel = "probability", ylabel = "TP rate",
         ratio = 1.0, xlims = (0,1), ylims = (0,1))
plot!(pl_lip, θs, θs)
θs = collect(0.0:0.01:1)
for i in 1:length(d_found)
    plot!(pl_lip, θs, tp[i][2], label = "d = $(d_found[i])")
end


# plot(pl_ww, pl_lip, layout=(1,2))

# Plot of TP for both
pl_tp = plot(title = "TP for Ww and HIPM", xlabel = "probability", ylabel = "TP rate",
          ratio = 1.0, xlims = (0,1), ylims = (0,1))

# Plot the diagonal reference line
plot!(pl_tp, θs, θs, color ="black")

for i in 1:length(d_found)
    # Store the color of the first plot
    # Get the default color used for `i`th plot
    c = palette(:auto)[i]
    # Plot the solid line for WW
    plot!(pl_tp, θs, tp[i][1], label = "d = $(d_found[i])", color = c)
    # Plot the dashed line with the same color for HIPM
    plot!(pl_tp, θs, tp[i][2], linestyle = :dash,  linewidth=1,color = c, label=false)
end


# Plot of FP for both
pl_fp = plot(title = "FP for Ww and HIPM", xlabel = "probability", ylabel = "FP rate",
          ratio = 1.0, xlims = (0,1), ylims = (0,1))
plot!(pl_fp, θs, θs, color ="black", label = false)
plot!(pl_fp, θs, fp[1], label = "WW")
plot!(pl_fp, θs, fp[2], linestyle = :dash,  linewidth=1, label = "HIPM")


# Plot of ROC for both
pl_roc = plot(title = "ROC for Ww and HIPM", xlabel = "False positive rate", ylabel = "True positive rate",
          ratio = 1.0, xlims = (0,1), ylims = (0,1))

for i in 1:length(d_found)
    # Get the default color used for `i`th plot
    c = palette(:auto)[i]
    # Plot the solid line for WW
    plot!(pl_roc, fp[1], tp[i][1], label = "d = $(d_found[i])", color = c)
    # Plot the dashed line with the same color for HIPM
    plot!(pl_roc, fp[2], tp[i][2], linestyle = :dash,  linewidth=1,color = c, label=false)
end
scatter!(pl_roc,[],[], label = "dash:HIMP")



        