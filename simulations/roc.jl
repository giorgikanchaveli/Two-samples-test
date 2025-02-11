include("../approaches/emp_threshold_approach.jl")
include("../approaches/permutation_approach.jl")
include("../distances/distance_Wasserstein.jl")
include("../distances/new_distance.jl")
include("../distances/w_distance.jl")
using QuadGK

function direct_sampling(p::PPM, q::PPM, n_top::Int, n_bottom::Int, nReps::Int)
    # Given two laws of RPM we simulate the distances between empirical measures 
    

    d_ww, d_lip = zeros(nReps), zeros(nReps)
    for i in 1:nReps
        if i % 10 == 0
            println("iteration (direct): $i")
        end
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
        if i % 10 == 0
            println("iteration (permutation): $i")
        end
        indices = randperm(p_emp.n + q_emp.n)

        p_emp_shuffled = emp_ppm(total_atoms[indices[1:p_emp.n],:], p_emp.n, p_emp.m , p_emp.a, p_emp.b)
        q_emp_shuffled = emp_ppm(total_atoms[indices[p_emp.n+1:end],:], q_emp.n, q_emp.m, q_emp.a, q_emp.b)

        d_ww[i] = ww(p_emp_shuffled, q_emp_shuffled)
        d_lip[i] = dlip(p_emp_shuffled, q_emp_shuffled)
    end
    return d_ww, d_lip
end




function rej_rates(p::PPM, q::PPM, n_top::Int, n_bottom::Int, nReps::Int)
    # this function computes the rejection rate of the permutation test
    # for the given PPMs p and q, for both distances

    # nReps : number of repetitions of the permutation test
    # n_top, n_bottom : number of top and bottom samples
    # p, q : PPMs
    # returns the rejection rate
    θs = collect(0.0:0.01:1)
    perm_par = 50

    # get distances
    d_ww, d_lip = direct_sampling(p, q, n_top, n_bottom, nReps)

    # get thresholds
    d_ww_perm, d_lip_perm = permuted_sampling(generate_emp(p, n_top, n_bottom), generate_emp(q, n_top, n_bottom), perm_par)
    thresh_ww = quantile(d_ww_perm, 1 .- θs)
    thresh_lip = quantile(d_lip_perm, 1 .- θs)
    
    # get rates
    rej_ww = [mean(d_ww .> t) for t in thresh_ww]
    rej_lip = [mean(d_lip .> t) for t in thresh_lip]

    return rej_ww, rej_lip
end

function plot_rej(r_ww::Vector{Float64}, r_lip::Vector{Float64}, title::String, labels::Vector{String})
    # this function plots the rejection rates for the two distances
    # r_ww, r_lip : rejection rates
    # labels : labels for the two distances
    θs = collect(0.0:0.01:1)
    pl = plot(title = title, xlabel = "probability", ylabel = "rejection rate", ratio = 1.0,xlims = (0,1), ylims = (0,1))
    plot!(pl, θs, r_ww, label = labels[1])
    plot!(pl, θs, r_lip, label = labels[2])
    return pl
end

function plot_roc(fp_rates::Vector{Vector{Float64}}, tp_rates::Vector{Vector{Float64}}, title::String, labels::Vector{String})
    # this function plots the ROC curve
    # fp_rates, tp_rates : false positive and true positive rates for each distane function
    # labels : labels for the two distances
    pl = plot(title = title, xlabel = "False positive rate", ylabel = "True positive rate", ratio = 1.0,xlims = (0,1), ylims = (0,1))
    plot!(pl, fp_rates[1], tp_rates[1], label = labels[1])
    plot!(pl, fp_rates[2], tp_rates[2], label = labels[2])
    return pl
end





function save_figures(measures::String, n_top::Int, n_bottom::Int, nReps::Int, save)
    # this function saves the figures for the given measures for fp, tp, roc
    # measures : the measures to compare
    # n_top, n_bottom : number of top and bottom samples
    # nReps : number of repetitions of the permutation test

    p_same = ()->probability("same")
    p_splitting = ()->probability("splitting")
    dp_same = DP(1.0, p_same, -1.0, 1.0)
    dp_splitting = DP(1.0, p_splitting, -1.0, 1.0)

    p_a, p_b, q_a, q_b = 1.2, 0.3, 1.1, 3.4
    p_beta_1 = ()->rand(Beta(p_a, p_b))
    p_betaclose = () -> rand(Beta(p_a, p_b + 0.1))
    p_betafaraway = ()->rand(Beta(q_a, q_b))
    dp_beta_1 = DP(1.0, p_beta_1, -1.0, 1.0)
    dp_betafaraway = DP(1.0, p_betafaraway, -1.0, 1.0)
    dp_betaclose = DP(1.0, p_betaclose, -1.0, 1.0)


    filepath = joinpath(pwd(), "plots/rejrates")
    if measures == "samesplitting"
        fp_ww, fp_lip = rej_rates(dp_same, dp_same, n_top, n_bottom, nReps)
        tp_ww, tp_lip = rej_rates(dp_same, dp_splitting, n_top, n_bottom, nReps)
        pl_fp = plot_rej(fp_ww, fp_lip, "FP, samesplitting", ["ww", "lip"])
        pl_tp = plot_rej(tp_ww, tp_lip, "TP, samesplitting", ["ww", "lip"])
        pl_roc = plot_roc([fp_ww, fp_lip],[tp_ww, tp_lip], "ROC, samesplitting", ["ww", "lip"])
        
    elseif measures == "betafaraway"
        fp_ww, fp_lip = rej_rates(dp_beta_1, dp_beta_1, n_top, n_bottom, nReps)
        tp_ww, tp_lip = rej_rates(dp_beta_1, dp_betafaraway, n_top, n_bottom, nReps)
        pl_fp = plot_rej(fp_ww, fp_lip, "FP, betafaraway", ["ww", "lip"])
        pl_tp = plot_rej(tp_ww, tp_lip, "TP, betafaraway", ["ww", "lip"])
        pl_roc = plot_roc([fp_ww, fp_lip],[tp_ww, tp_lip], "ROC, betafaraway", ["ww", "lip"])

    else

        fp_ww, fp_lip = rej_rates(dp_beta_1, dp_beta_1, n_top, n_bottom, nReps)
        tp_ww, tp_lip = rej_rates(dp_beta_1, dp_betaclose, n_top, n_bottom, nReps)
        pl_fp = plot_rej(fp_ww, fp_lip, "FP, betaclose", ["ww", "lip"])
        pl_tp = plot_rej(tp_ww, tp_lip, "TP, betaclose", ["ww", "lip"])
        pl_roc = plot_roc([fp_ww, fp_lip],[tp_ww, tp_lip], "ROC, betaclose", ["ww", "lip"])
    end
    filepath = joinpath(filepath, "$measures/n = $(n_top), m = $(n_bottom)")
    if save
        savefig(pl_fp, joinpath(filepath, "fp_$(measures)_$(n_top)_$(n_bottom).png"))
        savefig(pl_tp, joinpath(filepath, "tp_$(measures)_$(n_top)_$(n_bottom).png"))
        savefig(pl_roc, joinpath(filepath, "roc_$(measures)_$(n_top)_$(n_bottom).png"))
    end
 
end

function save_figures(n_top::Int, n_bottom::Int, nReps::Int, save)
    save_figures("samesplitting", n_top, n_bottom, nReps, save)
    println("finished samesplitting")
    save_figures("betafaraway", n_top, n_bottom, nReps, save)
    println("finished betafaraway")
    save_figures("betaclose", n_top, n_bottom, nReps, save)
end





# t = time()
# save_figures(16, 5000, 100, true)
# println("finished n = 16")
# save_figures(128, 5000, 100, true)
# t = time() - t
# p_same = ()->probability("same")
# p_splitting = ()->probability("splitting")
# dp_same = DP(1.0, p_same, -1.0, 1.0)
# dp_splitting = DP(1.0, p_splitting, -1.0, 1.0)

# p_a, p_b, q_a, q_b = 1.2, 0.3, 1.1, 3.4
# p_beta_1 = ()->rand(Beta(p_a, p_b))
# p_betaclose = () -> rand(Beta(p_a, p_b + 0.1))
# p_betafaraway = ()->rand(Beta(q_a, q_b))
# dp_beta_1 = DP(1.0, p_beta_1, -1.0, 1.0)
# dp_betafaraway = DP(1.0, p_betafaraway, -1.0, 1.0)
# dp_betaclose = DP(1.0, p_betaclose, -1.0, 1.0)

# filepath = joinpath(pwd(), "plots/rejrates")
# if measures == "samesplitting"
#     fp_ww, fp_lip = rej_rates(dp_same, dp_same, n_top, n_bottom, nReps)
#     tp_ww, tp_lip = rej_rates(dp_same, dp_splitting, n_top, n_bottom, nReps)
#     pl_fp = plot_rej(fp_ww, fp_lip, "FP, samesplitting", ["ww", "lip"])
#     pl_tp = plot_rej(tp_ww, tp_lip, "TP, samesplitting", ["ww", "lip"])
#     pl_roc = plot_roc([fp_ww, fp_lip],[tp_ww, tp_lip], "ROC, samesplitting", ["ww", "lip"])
    
# elseif measures == "betafaraway"
#     fp_ww, fp_lip = rej_rates(dp_beta_1, dp_beta_1, n_top, n_bottom, nReps)
#     tp_ww, tp_lip = rej_rates(dp_beta_1, dp_betafaraway, n_top, n_bottom, nReps)
#     pl_fp = plot_rej(fp_ww, fp_lip, "FP, betafaraway", ["ww", "lip"])
#     pl_tp = plot_rej(tp_ww, tp_lip, "TP, betafaraway", ["ww", "lip"])
#     pl_roc = plot_roc([fp_ww, fp_lip],[tp_ww, tp_lip], "ROC, betafaraway", ["ww", "lip"])

# else
#     fp_ww, fp_lip = rej_rates(dp_beta_1, dp_beta_1, n_top, n_bottom, nReps)
#     tp_ww, tp_lip = rej_rates(dp_beta_1, dp_betaclose, n_top, n_bottom, nReps)
#     pl_fp = plot_rej(fp_ww, fp_lip, "FP, betaclose", ["ww", "lip"])
#     pl_tp = plot_rej(tp_ww, tp_lip, "TP, betaclose", ["ww", "lip"])
#     pl_roc = plot_roc([fp_ww, fp_lip],[tp_ww, tp_lip], "ROC, betaclose", ["ww", "lip"])
# end
# filepath = joinpath(filepath, "$measures/n = $(n_top), m = $(n_bottom)")
# savefig(pl_fp, joinpath(filepath, "fp_$(measures)_$(n_top)_$(n_bottom).png"))
# savefig(pl_tp, joinpath(filepath, "tp_$(measures)_$(n_top)_$(n_bottom).png"))
# savefig(pl_roc, joinpath(filepath, "roc_$(measures)_$(n_top)_$(n_bottom).png"))






