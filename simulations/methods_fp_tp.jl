# We compare the testing schemes by comparing false positive and true positive rates. 
# In this file there are methods to use for simulations
include("../approaches/emp_threshold_approach.jl")
include("../approaches/permutation_approach.jl")
include("../distances/distance_Wasserstein.jl")
include("../distances/new_distance.jl")
include("../distances/w_distance.jl")
#include("approaches/threshold_approach.jl")

using Plots


function rej_rate(dist::Function, problem::param_problem_pm, par_emp::param_emp, iter::Int)
    # For i.i.d

    # estimate the rejection rate for the empirical threshold approach given probability 
    # measures p and q from problem parameters.
    @extract problem: p q n_rv
    
    θ = collect(0.0:0.01:1.0)
    thresholds = emp_thresholds(dist, par_emp, n_rv, θ) # estimate the thresholds for the distance between p_0 and q_0 
                                                   # from parameters of the empirical approach
    r = zeros(length(θ))
    test_stat = sqrt(n_rv / 2) * sample_distances(dist, p, q, n_rv, iter)
    for i in 1:length(θ)
        r[i] = sum(test_stat .> thresholds[i]) # how often do we reject H_0
    end
    r = r / iter
    return r
end

function rej_rate(dist::Function, problem::param_problem_pm, par_perm::param_perm, iter::Int, seed = 1234)
    # For i.i.d
    
    # estimate the rejection rate for the permutation approach given probability 
    # measures p and q from problem parameters.
    @extract problem: p q n_rv
    
    θ = collect(0.0:0.01:1.0)
    Random.seed!(seed)
    p_emp, q_emp = generate_emp(p, n_rv), generate_emp(q, n_rv)
    thresholds = perm_thresholds(dist, p_emp, q_emp, par_perm, θ, seed)
    test_stats = sqrt(n_rv / 2) * sample_distances(dist, p, q, n_rv, iter, seed + 1234) # I don't want
                                # seed to coincide with the seed for generating empirical measures from which   
                                # we get permutation threshold
    r = [mean(test_stats .> t) for t in thresholds] # rejection rates per each probability level

    return r
end



# Exchangeable

function rej_rate(dist::Function, problem::param_problem_ppm, par_emp::param_emp_ppm, iter::Int,seed = 1234)
    # For exchangeable

    # estimate the rejection rate for the empirical threshold approach given probability 
    # measures p and q from problem parameters.
    @extract problem: p q n m
    Random.seed!(seed)
    θ = collect(0.0:0.01:1.0)
    thresholds = emp_thresholds(dist, par_emp, n, m, θ) # estimate the thresholds for the distance between p_0 and q_0 
                                                   # from parameters of the empirical approach
    r = zeros(length(θ))
    test_stats = sqrt(n / 2) * sample_distances(dist, p, q, n, m, iter)
    r = [mean(test_stats .> t) for t in thresholds]
    return r
end




function rej_rate(dist::Function, problem::param_problem_ppm, par_perm::param_perm, iter::Int, seed = 1234)
    # For Exchangeable
    
    # estimate the rejection rate for the permutation approach given probability 
    # measures p and q from problem parameters.
    @extract problem: p q n m
    
    θ = collect(0.0:0.01:1.0)
    Random.seed!(seed)
    p_emp, q_emp = generate_emp(p, n, m), generate_emp(q, n, m)
    thresholds = perm_thresholds(dist, p_emp, q_emp, par_perm, θ, seed) #seed may not be necesarry here
    test_stats = sqrt(n / 2) * sample_distances(dist, p, q, n, m, iter, seed + 1234)
    r = [mean(test_stats .> t) for t in thresholds]

    return r
end



function roc(α, β, strs::Array{String}, t::String)
    # plot the ROC curve
    p = plot()
    for i in 1:length(α)
        plot!(p, α[i], β[i], ratio = 1, xlims = (0,1), ylims = (0,1), xlabel="false positive rate", 
        ylabel="true positive rate", title=t, label = strs[i])
    end
    return p
end

