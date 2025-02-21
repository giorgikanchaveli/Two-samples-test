include("simulation_functions.jl")


# This file is used to compare the thresholds and rejection rates of the pooled measure and the permutation method


function produce_plots(d::Float64, n_top::Int64, n_bottom::Int64)
    p_a, p_b, q_a, q_b = find_betas(d)
    p_1 = ()->rand(Beta(p_a, p_b))
    p_2 = ()->rand(Beta(q_a, q_b))
    α = 1.0
    dp_1 = DP(α, p_1, 0.0, 1.0)
    dp_2 = DP(α, p_2, 0.0, 1.0)

    # compute distances using direct samples
    nReps = 100 
    d_ww, d_lip = direct_sampling(dp_1, dp_2, n_top, n_bottom, nReps)
    # get thresholds
    nPerms = 75 
    d_ww_pooled, d_lip_pooled = pooled_measure_sampling(dp_1, dp_2, n_top, n_bottom, nPerms)
    d_ww_perm, d_lip_perm = permuted_sampling(generate_emp(dp_1, n_top, n_bottom), 
                                    generate_emp(dp_2, n_top, n_bottom), nPerms)


    θs = collect(0.0:0.01:1.0)
    thresh_pooled_ww, thresh_pooled_lip = quantile(d_ww_pooled, 1 .- θs), quantile(d_lip_pooled, 1 .- θs)
    thresh_perm_ww, thresh_perm_lip = quantile(d_ww_perm, 1 .- θs), quantile(d_lip_perm, 1 .- θs)
    quant_ww, quant_lip = quantile(d_ww, 1 .- θs), quantile(d_lip, 1 .- θs)

    # rejection rates
    rej_pooled_ww = [mean(d_ww .> t) for t in thresh_pooled_ww]
    rej_pooled_lip = [mean(d_lip .> t) for t in thresh_pooled_lip]
    rej_perm_ww = [mean(d_ww .> t) for t in thresh_perm_ww]
    rej_perm_lip = [mean(d_lip .> t) for t in thresh_perm_lip]




    # Plots
    pl_thresholds = plot_vectors([thresh_pooled_ww, thresh_pooled_lip, thresh_perm_ww, thresh_perm_lip], "thresholds", 
                    ["ww pooled", "lip pooled", "ww perm", "lip perm"])

    pl_thresholds_ww = plot_vectors([thresh_pooled_ww, thresh_perm_ww], "ww thresholds", ["pooled", "perm"])
    pl_thresholds_lip = plot_vectors([thresh_pooled_lip, thresh_perm_lip], "lip thresholds", ["pooled", "perm"])

    pl_obs_thresh_pooled = plot_vectors([quant_ww, quant_lip, thresh_pooled_ww, thresh_pooled_lip], "observations and thresholds from pooling", 
                    ["ww quantiles", "lip quantiles", "ww threshold", "lip threshold"])
    pl_obs_thresh_perm = plot_vectors([quant_ww, quant_lip, thresh_perm_ww, thresh_perm_lip], "observations and thresholds from permutation", 
                    ["ww quantiles", "lip quantiles", "ww threshold", "lip threshold"])
    pl_thresh_rej_pooled = plot_vectors([thresh_pooled_ww, thresh_pooled_lip, rej_pooled_ww, rej_pooled_lip], "thresholds and rejection rates using pooled", 
                    ["ww threshold", "lip threshold", "ww rejection", "lip rejection"])
    pl_thresh_rej_perm = plot_vectors([thresh_perm_ww, thresh_perm_lip, rej_perm_ww, rej_perm_lip], "thresholds and rejection rates using permutation", 
                    ["ww threshold", "lip threshold", "ww rejection", "lip rejection"])

    pl_rejections_ww = plot_vectors([rej_pooled_ww, rej_perm_ww], "rejection rates for ww", ["pooled", "perm"])
    pl_rejections_lip = plot_vectors([rej_pooled_lip, rej_perm_lip], "rejection rates for lip", ["pooled", "perm"])

    # return all plots in list
    return [pl_thresholds, pl_thresholds_ww, pl_thresholds_lip, 
            pl_obs_thresh_pooled, pl_obs_thresh_perm, pl_thresh_rej_pooled, pl_thresh_rej_perm, pl_rejections_ww, pl_rejections_lip]
end

function savingfigures(figure, d::Float64)
    filepath = joinpath(pwd(), "plots/pooled_vs_permutation")
    savefig(figure[2], joinpath(filepath, "thresholds_ww_$(d).png"))
    savefig(figure[3], joinpath(filepath, "thresholds_lip_$(d).png"))
    savefig(figure[8], joinpath(filepath, "rejections_ww_$(d).png"))
    savefig(figure[9], joinpath(filepath, "rejections_lip_$(d).png"))
end



figures_1 = produce_plots(0.01, 80, 5000)
figures_2 = produce_plots(0.1, 80, 5000)
figures_3 = produce_plots(0.3, 80, 5000)
figures_4 = produce_plots(0.5, 80, 5000)

# save figures

savefigures(figures_1, 0.01)
savefigures(figures_2, 0.1)
savefigures(figures_3, 0.3)
savefigures(figures_4, 0.5)




