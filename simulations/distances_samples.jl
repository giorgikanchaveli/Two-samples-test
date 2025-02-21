include("../approaches/emp_threshold_approach.jl")
include("../approaches/permutation_approach.jl")
include("../distances/distance_Wasserstein.jl")
include("../distances/new_distance.jl")
include("../distances/w_distance.jl")
using QuadGK


function int_wass_beta(p_a, p_b, q_a, q_b)
    # this function computes the Wasserstein distance between two Beta distributions
    # with parameters p and q using numerical integration
    p = Beta(p_a, p_b)
    q = Beta(q_a, q_b)
    cdf_p = x -> cdf(p, x)
    cdf_q = x -> cdf(q, x)
    f = x -> abs(cdf_p(x) - cdf_q(x))
    return quadgk(f, 0, 1)[1]
end

# methods for sampling
function direct_sampling(p::PPM, q::PPM, n_top::Int, n_bottom::Int, nReps::Int)
    # Given two laws of RPM we simulate the distances between empirical measures 
  

    d_ww, d_lip = zeros(nReps), zeros(nReps)
    for i in 1:nReps
        if i % 10 == 0
            println("iteration (direct): $i")
        end
        p_emp, q_emp = generate_emp(p, n_top, n_bottom), generate_emp(q, n_top, n_bottom)
        d_ww[i] = ww(p_emp, q_emp) # copy(p_emp) isn't necessary
        d_lip[i] = dlip(p_emp, q_emp) # copy(p_emp) isn't necessary
    end
    return d_ww, d_lip
end

function permuted_sampling(p_emp::emp_ppm, q_emp::emp_ppm, nReps::Int)
    
    # Given two empirical measures we permute them and compute distances
    d_ww, d_lip = zeros(nReps), zeros(nReps)
    total_atoms = vcat(p_emp.atoms, q_emp.atoms)
    for i in 1:nReps
        if i % 10 == 0
            println("iteration (permutation): $i")
        end
        indices = randperm(p_emp.n + q_emp.n)

        p_emp_shuffled = emp_ppm(total_atoms[indices[1:p_emp.n],:], p_emp.n, p_emp.m , p_emp.a, p_emp.b)
        q_emp_shuffled = emp_ppm(total_atoms[indices[p_emp.n+1:end],:], q_emp.n, q_emp.m, q_emp.a, q_emp.b)

        d_ww[i] = ww(p_emp_shuffled, q_emp_shuffled) # copy(p_emp_shuffed) isn't necessary
        d_lip[i] = dlip(p_emp_shuffled, q_emp_shuffled) # copy(p_emp_shuffed) isn't necessary
    end
    return d_ww, d_lip
end

# p_a, p_b, q_a, q_b = 1.2, 0.3, 1.1, 3.4
# p_1 = ()->rand(Beta(p_a, p_b))
# p_2 = ()->rand(Beta(p_a, p_b + .2))
# d_true = round(int_wass_beta(p_a, p_b, q_a, q_b), digits = 6)


# methods for plotting histograms

function plot_hist(d_ww, d_lip, title)
    x_max = maximum([maximum(d_ww), maximum(d_lip)]) # sets the limit of the x-axis 
    #x_min = minimum([minimum(d_ww), minimum(d_lip)])
    
    h_ww = histogram(d_ww, label="ww", xlabel="distance", ylabel="frequency",
                    title=title,
                    xticks=0.0:0.1:1.3, xlims = (0, x_max),bins = 30)
    vline!(h_ww, [mean(d_ww)], label="mean = $(round(mean(d_ww),digits = 5))", color="red")

    h_lip = histogram(d_lip, label="lip", xlabel="distance", ylabel="frequency",
                    title=title,
                    xticks=0.0:0.1:1.3, xlims = (0, x_max), bins = 30)
        
    vline!(h_lip, [mean(d_lip)], label="mean = $(round(mean(d_lip), digits = 5))", color="red")
    return h_ww, h_lip
end


function plot_vectors(v::Vector{Vector{Float64}}, title, labels)
    # plot the vectors
    θs = collect(0.0:0.01:1.0)
    p = plot(title = title, xlabel = "probability", ylabel = "values", ratio = 1.0,xlims = (0,1), ylims = (0,1))
    for i in 1:length(v)
        plot!(p, θs, v[i], label = labels[i])
    end
    return p
end

function figures(measures::String, n_top::Int, n_bottom::Int, nReps::Int)
    α = 1.0
    if measures == "samesplitting"
        p_1 = ()->probability("same") # uniform(-1/2, 1/2)
        p_2 = ()->probability("splitting") # 1/2 uniform(-1, -3/4) + 1/2 uniform(3/4, 1)
        dp_1 = DP(α, p_1, -1.0, 1.0)
        dp_2 = DP(α, p_2, -1.0, 1.0)
        #d_true = quadgk(x -> abs(cdf_same(x) - cdf_splitting(x)), -1, 1)[1]
    elseif measures == "betafaraway" # both distances = 0.556
        p_a, p_b, q_a, q_b = 1.2, 0.3, 1.1, 3.4
        p_1 = ()->rand(Beta(p_a, p_b))
        p_2 = ()->rand(Beta(q_a, q_b))
        dp_1 = DP(α, p_1, 0.0, 1.0)
        dp_2 = DP(α, p_2, 0.0, 1.0)
        #d_true = round(int_wass_beta(p_a, p_b, q_a, q_b), digits = 6)
    else
        p_a, p_b, q_a, q_b = 1.2, 1.2, 1.2, 1.25 # both distances = 0.01
        p_1 = ()->rand(Beta(p_a, p_b))
        p_2 = ()->rand(Beta(q_a, q_b))
        dp_1 = DP(α, p_1, 0.0, 1.0)
        dp_2 = DP(α, p_2, 0.0, 1.0)
        #d_true = round(int_wass_beta(p_a, p_b, q_a, q_b), digits = 6)
    end

    

    # compute distances using direct samples
    d_ww, d_lip = direct_sampling(dp_1, dp_2, n_top, n_bottom, nReps)
    # plot histograms, quantiles
    h_ww, h_lip = plot_hist(d_ww, d_lip, "direct, $(measures)")
    h = plot(h_ww, h_lip, layout=(2,1), link = :x)
    d_ww_quantiles = quantile(d_ww, 1 .- θs)
    d_lip_quantiles = quantile(d_lip, 1 .- θs)
    q_plot = plot_vectors([d_ww_quantiles, d_lip_quantiles],"direct, $(measures)", ["ww", "lip"])

    # get thresholds and data quantiles
    perm_par = 24
    θs = collect(0.0:0.01:1.0)
    d_ww_perm, d_lip_perm = permuted_sampling(generate_emp(dp_1, n_top, n_bottom), generate_emp(dp_2, n_top, n_bottom), perm_par)
    thresh_ww = quantile(d_ww_perm, 1 .- θs)
    thresh_lip = quantile(d_lip_perm, 1 .- θs)
    d_ww_quantiles = quantile(d_ww, 1 .- θs)
    d_lip_quantiles = quantile(d_lip, 1 .- θs)
    thresh_quant_plots = plot_vectors([thresh_ww, thresh_lip, d_ww_quantiles, d_lip_quantiles], 
                                "thresholds, $(measures)", ["ww", "lip", "ww quantiles", "lip quantiles"])

    # add rejection rates to thresholds
    rej_ww = [mean(d_ww .> t) for t in thresh_ww]
    rej_lip = [mean(d_lip .> t) for t in thresh_lip]
    thresh_rej_plots = plot_vectors([thresh_ww, thresh_lip, rej_ww, rej_lip],
                     "thresholds and rejection rates, $(measures)", ["ww", "lip", "ww rejection", "lip rejection"])
    return h, q_plot, thresh_quant_plots, thresh_rej_plots
end

function save_figures(measures::String, n_top::Int, n_bottom::Int, nReps::Int)
    h, q_plot, thresh_quant_plots, thresh_rej_plots = figures(measures, n_top, n_bottom, nReps)
    filepath = joinpath(pwd(), "plots/samples,rejrates")
    savefig(h, joinpath(filepath, "$measures/n = $(n_top), m = $n_bottom/h_$(measures)_$(n_top)_$n_bottom.png"))
    savefig(q_plot, joinpath(filepath, "$measures/n = $(n_top), m = $n_bottom/q_plot_$(measures)_$(n_top)_$(n_bottom).png"))
    savefig(thresh_quant_plots, joinpath(filepath, "$measures/n = $(n_top), m = $(n_bottom)/thresh_quant_plots_$(measures)_$(n_top)_$(n_bottom).png"))
    savefig(thresh_rej_plots, joinpath(filepath, "$measures/n = $(n_top), m = $n_bottom/thresh_rej_plots_$(measures)_$(n_top)_$(n_bottom).png"))
end

function save_figures(n_top::Int, n_bottom::Int, nReps::Int)
    save_figures("samesplitting", n_top, n_bottom, nReps)
    println("finished samesplitting")
    save_figures("betafaraway", n_top, n_bottom, nReps)
    println("finished betafaraway")
    save_figures("betaclose", n_top, n_bottom, nReps)
end

# check the output of save_fig

#h, q_plot, thresh_quant_plots, thresh_rej_plots = figures("betaclose", 16, 5000, 24)
# t = time()
# save_figures(16, 5000, 24)
# println("finished n = 16")
# save_figures(128, 5000, 24)
# t = time() - t
# Define measures

# p_1 = ()->probability("same")
# p_2 = ()->probability("splitting")
# dp_1 = DP(1.0, p_1, -1.0, 1.0)
# dp_2 = DP(1.0, p_2, -1.0, 1.0)
# n_top, n_bottom = 16, 5000

# # compute distances using direct samples
# nReps = 24
# d_ww, d_lip = direct_sampling(dp_1, dp_2, n_top, n_bottom, nReps)

# h_ww, h_lip = plot_hist(d_ww, d_lip, "direct, samesplitting")
# h = plot(h_ww, h_lip, layout=(2,1), link = :x)
# d_ww_quantiles = quantile(d_ww, θs)
# d_lip_quantiles = quantile(d_lip, θs)
# q_plot = plot_vectors([d_ww_quantiles,d_lip_quantiles], "direct, samesplitting", ["ww", "lip"])

# θs = collect(0.0:0.01:1.0)
# t = time()
# d_ww_perm, d_lip_perm = permuted_sampling(generate_emp(dp_1, n_top, n_bottom), generate_emp(dp_2, n_top, n_bottom), 100)
# t = time() - t
# thresh_ww = quantile(d_ww_perm, 1 .- θs)
# thresh_lip = quantile(d_lip_perm, 1 .- θs)

# thresh_plots = plot_vectors([thresh_ww, thresh_lip], "thresholds, samesplitting", ["ww", "lip"])
# d_ww_quantiles = quantile(d_ww, θs)
# d_lip_quantiles = quantile(d_lip, θs)
# rej_ww = [mean(d_ww .> t) for t in thresh_ww]
# rej_lip = [mean(d_lip .> t) for t in thresh_lip]

# thresh_rej_plots = plot_vectors([thresh_ww, thresh_lip, rej_ww, rej_lip],
#                      "thresholds and rejection rates, samesplitting", ["ww", "lip", "ww rejection", "lip rejection"])

# thresh_quant_plots = plot_vectors([thresh_ww, thresh_lip, d_ww_quantiles, d_lip_quantiles],
#                      "thresholds and quantiles, samesplitting", ["ww", "lip", "ww quantiles", "lip quantiles"])
# # using above functions generate the plots of intereset
