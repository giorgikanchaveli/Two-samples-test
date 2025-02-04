include("../approaches/emp_threshold_approach.jl")
include("../approaches/permutation_approach.jl")
include("../distances/distance_Wasserstein.jl")
include("../distances/new_distance.jl")
include("../distances/w_distance.jl")

function plot_hist(t_ww, t_lip, str)
    x_max = maximum([maximum(t_ww), maximum(t_lip)]) # sets the limit of the x-axis 
    
    h_ww = histogram(t_ww, label="ww", xlabel="distance", ylabel="frequency",
                    title="Histogram of distances between $(str[1]) measures",
                    xticks=0.0:0.2:3.0, xlims = (0, x_max),bins = 30)
    vline!(h_ww, [mean(t_ww)], label="mean", color="red")

    h_lip = histogram(t_lip, label="lip", xlabel="distance", ylabel="frequency",
                    title="Histogram of distances between $(str[2]) empirical measures",
                    xticks=0.0:0.2:3.0, xlims = (0, x_max), bins = 30)
        
    vline!(h_lip, [mean(t_lip)], label="mean", color="red")
    return h_ww, h_lip
end

function plot_quantiles(values, str)
    θs = collect(0.0:0.01:1.0)
    q_plot = plot(title="Q plot",)
    for i in 1:length(str)
        plot!(q_plot, θs, values[i], xlabel="probability", ylabel="quantiles", label = str[i])
    end
    return q_plot
end


function plot_quantiles_exp(values, str)
    θs = collect(0.0:0.01:1.0)
    q_plot = plot(title="Q plot",)
    for i in 1:length(str)
        plot!(q_plot, θs, exp.(values[i]), xlabel="probability", ylabel="quantiles", label = str[i])
    end
    return q_plot
end
# Define measures

# p_1 = ()->probability("same")
# p_2 = ()->probability("splitting")
# dp_1 = DP(1.0, p_1, -1.0, 1.0)
# dp_2 = DP(1.0, p_2, -1.0, 1.0)
p_a, p_b = 2.5, 4.0
q_a, q_b = 2.5, 4.0
beta_p, beta_q = Beta(p_a, p_b), Beta(q_a, q_b)
p_1 = ()->rand(beta_p)
p_2 = ()->rand(beta_q)
dp_1 = DP(2.0, p_1, -1.0, 1.0)
dp_2 = DP(1.0, p_1, -1.0, 1.0)
n_top, n_bottom = 100, 5000

# compute distances using direct samples
seed = 4567
s = 200
t = time()


t_ww_direct = sqrt(n_top/2)*sample_distances(ww, dp_1, dp_2, n_top, n_bottom, s)
t_lip_direct = sqrt(n_top/2)*sample_distances(dlip, dp_1, dp_2, n_top, n_bottom, s)

# compute distances using permutation
p_emp, q_emp = generate_emp(dp_1, n_top, n_bottom), generate_emp(dp_2, n_top, n_bottom)


t_ww_perm = sqrt(n_top/2)*sample_distances(ww, p_emp, q_emp, param_perm(s), seed)
t_lip_perm = sqrt(n_top/2)*sample_distances(dlip, p_emp, q_emp, param_perm(s), seed)


h_ww_direct, h_lip_direct = plot_hist(t_ww_direct, t_lip_direct, ["direct", "direct"])
h_ww_perm, h_lip_perm = plot_hist(t_ww_perm, t_lip_perm, ["permuted", "permuted"])


p_direct = plot(h_ww_direct, h_lip_direct, layout=(2,1), link = :x)
p_perm = plot(h_ww_perm, h_lip_perm, layout=(2,1), link = :x)

filepath = joinpath(pwd(),"plots/n = $(n_top), m = $(n_bottom)")
savefig(p_direct, joinpath(filepath, "hist_direct_ww_lip_$(n_top)_$(n_bottom)"))
savefig(p_perm, joinpath(filepath, "hist_perm_ww_lip_$(n_top)_$(n_bottom)"))



# compute thresholds
t = time() - t
θ = collect(0.0:0.01:1.0)
quant_dir_ww = quantile(t_ww_direct, θ)
quant_dir_lip = quantile(t_lip_direct, θ)
quant_dir = plot(title = "Quantiles of direct samples", xlabel = "probability", ylabel = "quantiles")
plot!(quant_dir, θ, quant_dir_ww, label = "ww", color = "red")
plot!(quant_dir, θ, quant_dir_lip, label = "lip", color = "blue")

quant_perm_ww = quantile(t_ww_perm, θ)
quant_perm_lip = quantile(t_lip_perm, θ)
quant_perm = plot(title = "Quantiles of permuted samples", xlabel = "probability", ylabel = "quantiles")
plot!(quant_perm, θ, quant_perm_ww, label = "ww", color = "red")
plot!(quant_perm, θ, quant_perm_lip, label = "lip", color = "blue")

savefig(quant_dir, joinpath(filepath, "quantiles_direct_$(n_top)_$(n_bottom)"))
savefig(quant_perm, joinpath(filepath, "quantiles_perm_$(n_top)_$(n_bottom)"))



thresholds_ww = quantile(t_ww_perm, 1 .- θ)
thresholds_lip = quantile(t_lip_perm, 1 .-θ)

thresholds_ww_direct = quantile(t_ww_direct, 1 .- θ)
thresholds_lip_direct = quantile(t_lip_direct, 1 .-θ)


rej_ww = [mean(t_ww_direct .> t) for t in thresholds_ww]
rej_lip = [mean(t_lip_direct .> t) for t in thresholds_lip]

thresholds_pl = plot_quantiles([thresholds_ww, thresholds_lip, thresholds_ww_direct, thresholds_lip_direct],
             ["ww_threshold", "dlip_threshold", "ww_quantiles", "dlip_quantiles"])
annotate!(thresholds_pl, θ[50], thresholds_ww[50], text("thresholds_ww", :black, :left, 8))
annotate!(thresholds_pl, θ[50], thresholds_lip[50], text("thresholds_lip", :black, :left, 8))
annotate!(thresholds_pl, θ[50], thresholds_ww_direct[50], text("ww_quant", :black, :left, 8))
annotate!(thresholds_pl, θ[50], thresholds_lip_direct[50], text("lip_quant", :black, :left, 8))

thresholds_pl_with_rej = deepcopy(thresholds_pl)


plot!(thresholds_pl_with_rej, θ, rej_ww, label = "rej_ww")
plot!(thresholds_pl_with_rej, θ, rej_lip, label = "rej_lip")
annotate!(thresholds_pl_with_rej, θ[6], rej_ww[6], text("rej_ww", :black, :left, 8))
annotate!(thresholds_pl_with_rej, θ[60], rej_lip[60], text("rej_lip", :black, :left, 8))

# plot!(thresholds_pl_exp, θ, rej_ww, label = "rej_ww")
# plot!(thresholds_pl_exp, θ, rej_lip, label = "rej_lip")
savefig(thresholds_pl, joinpath(filepath, "thresholds_$(n_top)_$(n_bottom)"))
savefig(thresholds_pl_with_rej, joinpath(filepath, "thresholds_with_rej$(n_top)_$(n_bottom)"))
# x_max = maximum([maximum(t_ww_direct), maximum(t_lip_direct)]) # sets the limit of the x-axis 
    
# h_ww_direct = histogram(t_ww_direct, label="ww", xlabel="distance", ylabel="frequency",
#                 title="Histogram of distances between empirical measures",
#                 xticks=0.0:0.2:3.0, xlims = (0, x_max),bins = 30)
# vline!(h_ww_direct, [mean(t_ww_direct)], label="mean", color="red")

# h_lip_direct = histogram(t_lip_direct, label="lip", xlabel="distance", ylabel="frequency",
#                 title="Histogram of distances between permuted empirical measures",
#                 xticks=0.0:0.2:3.0, xlims = (0, x_max), bins = 30)
    
# vline!(h_lip_direct, [mean(t_lip_direct)], label="mean", color="red")

# plot(h_ww_direct, h_lip_direct, layout=(2,1), link = :x)

