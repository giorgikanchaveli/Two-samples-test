include("../approaches/emp_threshold_approach.jl")
include("../approaches/permutation_approach.jl")
include("../distances/distance_Wasserstein.jl")
include("../distances/new_distance.jl")
include("../distances/w_distance.jl")
using QuadGK


# methods for sampling
function direct_sampling(p::PPM, q::PPM, n_top::Int, n_bottom::Int, nReps::Int, seed = 1234)
    # Given two laws of RPM we simulate the distances between empirical measures 
    Random.seed!(seed)

    d_ww, d_lip = zeros(nReps), zeros(nReps)
    for i in 1:nReps
        p_emp, q_emp = generate_emp(p, n_top, n_bottom), generate_emp(q, n_top, n_bottom)
        d_ww[i] = ww(p_emp, q_emp)
        d_lip[i] = dlip(p_emp, q_emp)
    end
    return d_ww, d_lip
end

function permuted_sampling(p_emp::emp_ppm, q_emp::emp_ppm, nReps::Int)
    # Given two empirical measures we permute them and compute distances
    d_ww, d_lip = zeros(nReps), zeros(nReps)
    total_atoms = vcat(p_emp.atoms, q_emp.atoms)
    for i in 1:nReps
        indices = randperm(p_emp.n + q_emp.n)

        p_emp_shuffled = emp_ppm(total_atoms[indices[1:p_emp.n],:], p_emp.n, p_emp.m , p_emp.a, p_emp.b)
        q_emp_shuffled = emp_ppm(total_atoms[indices[p_emp.n+1:end],:], q_emp.n, q_emp.m, q_emp.a, q_emp.b)

        d_ww[i] = ww(p_emp_shuffled, q_emp_shuffled)
        d_lip[i] = dlip(p_emp_shuffled, q_emp_shuffled)
    end
    return d_ww, d_lip
end


# methods for plotting histograms

function plot_hist(d_ww, d_lip, str)
    x_max = maximum([maximum(d_ww), maximum(d_lip)]) # sets the limit of the x-axis 
    x_min = minimum([minimum(d_ww), minimum(d_lip)])
    
    h_ww = histogram(d_ww, label="ww", xlabel="distance", ylabel="frequency",
                    title=str,
                    xticks=0.0:0.1:x_max, xlims = (0, x_max),bins = 30)
    vline!(h_ww, [mean(d_ww)], label="mean = $(round(mean(d_ww),digits = 5))", color="red")

    h_lip = histogram(d_lip, label="lip", xlabel="distance", ylabel="frequency",
                    title=str,
                    xticks=0.0:0.1:x_max, xlims = (0, x_max), bins = 30)
        
    vline!(h_lip, [mean(d_lip)], label="mean = $(round(mean(d_lip), digits = 5))", color="red")
    return h_ww, h_lip
end

function plot_quantiles(d_ww, d_lip, title, labels)
    θs = collect(0.0:0.01:1.0)
    q_plot = plot(title=title,)
    plot!(q_plot, θs, quantile(d_ww, θs), xlabel="probability", ylabel="quantiles", label = labels[1])
    plot!(q_plot, θs, quantile(d_lip, θs), xlabel="probability", ylabel="quantiles", label = labels[2])
    return q_plot
end






# Define measures

p_1 = ()->probability("same")
p_2 = ()->probability("splitting")
dp_1 = DP(1.0, p_1, -1.0, 1.0)
dp_2 = DP(1.0, p_2, -1.0, 1.0)
n_top, n_bottom = 16, 2

# compute distances using direct samples
nreps = 24
d_ww, d_lip = direct_sampling(dp_1, dp_2, n_top, n_bottom, nreps)

h_ww, h_lip = plot_hist(d_ww, d_lip, "direct, samesplitting")
h = plot(h_ww, h_lip, layout=(2,1), link = :x)
q_plot = plot_quantiles(d_ww, d_lip, "direct, samesplitting", ["ww", "lip"])


# using above functions generate the plots of intereset