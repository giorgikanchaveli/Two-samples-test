
# empirical approach for two sample hypothesis testing: Given two same p.m. we estimate the quantile of the distances and use that estimate
#                                                       for hypothesis testing.

using Distributions, Random
using ExtractMacro
using Plots

include("../distributions.jl")


# IID

function sample_distances(dist::Function, p::PM, q::PM, n::Int, s::Int, seed = 1234)
    # samples distances between empirical measures. Empirical measures are directly sampled
    d = zeros(s) # distances recorded for each simulated measures
    Random.seed!(seed)
    seeds = [rand(1:1000000) for i in 1:s] # this is needed to generate same empirical measures for making
                        # comparison between iid and exchangeable case. Because dlip is changing the seed
    for i in 1:s
        if i % 100 == 0
            println("iteration (iid, direct): $i")
        end
        # simulate measures
        Random.seed!(seeds[i]) # I need it because dlip is changing the seed
        p_emp, q_emp = generate_emp(p, n), generate_emp(q, n)
        d[i] = dist(p_emp, q_emp)
    end
    return d
end



function emp_thresholds(dist::Function, par_emp::param_emp, n_rv::Int, θ::Vector{Float64})
    # estimate the empirical thresholds for the distance between p and q for each θ
    @extract par_emp: p_0 s
    test_stat_samples = sort(sqrt(n_rv/2)*sample_distances(dist, p_0, p_0, n_rv, s))
    return quantile(test_stat_samples, 1 .- θ, sorted = true)
end





# Exchangeable



function sample_distances(dist::Function, p::PPM, q::PPM, n::Int, m::Int, s::Int, seed = 1234)
    # generate s samples of distance between empirical measures with n_rv random variables 
    # from p and q 
    distances = zeros(s)
    Random.seed!(seed)
    seeds = [rand(1:1000000) for i in 1:s] # this is needed to generate same empirical measures for making
             # comparison between iid and exchangeable case. Because dlip is changing the seed
    for i in 1:s
        if i%10 == 0
            println("iteration (exch, direct): $i")
        end
        Random.seed!(seeds[i]) # need it because dlip changes seed;  I want empirical measures from 
                               # iid and exch. case to have same atoms if m = 1
        emp_p, emp_q = generate_emp(p, n, m), generate_emp(q, n, m) 

        distances[i] = dist(emp_p, emp_q) 
    end
    return distances
end

function emp_thresholds(dist::Function, par_emp::param_emp_ppm, n::Int, m::Int, θ::Vector{Float64})
    # estimate the empirical thresholds for the distance between p and q for each θ
    @extract par_emp: p_aux s
    test_stat_samples = sort(sqrt(n/2)*sample_distances(dist, p_aux, p_aux, n, m, s))
    return quantile(test_stat_samples, 1 .- θ, sorted = true)
end






