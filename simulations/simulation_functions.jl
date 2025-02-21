include("../distances/distance_Wasserstein.jl")
include("../distances/new_distance.jl")
include("../distances/w_distance.jl")
using QuadGK


# functions for wasserstein distance betweeen beta distributions
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


function find_betas(d::Float64)
    # Given a value d, find two beta distributions with wass distance between them close to d
    # First beta distribution is fixed, for the second, only the parameter a is fixed

    # if found parameters, returns them, otherwise returns false
    # can find parameters for the distances in (0.0, 0.7)
    p_a, p_b = 1.0, 0.3
    a_range = collect(0.1:0.1:2.0)
    b_range = collect(0.1:0.1:2.0)
    ϵ = 0.01
    for a in a_range
        for b in b_range
            d_w = int_wass_beta(p_a, p_b, a, b)
            if abs(d_w - d) < ϵ
                return p_a, p_b, a, b
            end
        end
    end
    return -1.0,-1.0,-1.0,-1.0
end



# functions for sampling distances between two laws of RPM

function direct_sampling(p::PPM, q::PPM, n_top::Int, n_bottom::Int, nReps::Int)
    # Given two laws of RPM we simulate empirical measures and then compute 2 distances (ww, hipm) between them

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
    # Given two empirical measures we permute them and compute 2 distances (ww, hipm) between them
    
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






function rej_rates(p::PPM, q::PPM, n_top::Int, n_bottom::Int, nPerms::Int, nReps::Int)
    # this function computes the rejection rate of the permutation test
    # for the given PPMs p and q, for both distances

    # nReps : number of repetitions of the permutation test
    # n_top, n_bottom : number of top and bottom samples
    # p, q : PPMs
    # returns the rejection rate
    θs = collect(0.0:0.01:1)
    

    # get distances
    d_ww, d_lip = direct_sampling(p, q, n_top, n_bottom, nReps)

    # get thresholds
    d_ww_perm, d_lip_perm = permuted_sampling(generate_emp(p, n_top, n_bottom), 
                                generate_emp(q, n_top, n_bottom), nPerms)
    thresh_ww = quantile(d_ww_perm, 1 .- θs)
    thresh_lip = quantile(d_lip_perm, 1 .- θs)
    
    # get rates
    rej_ww = [mean(d_ww .> t) for t in thresh_ww]
    rej_lip = [mean(d_lip .> t) for t in thresh_lip]

    return rej_ww, rej_lip
end

function fp_rate(n_top::Int, n_bottom::Int, nPerms::Int, nReps::Int)
    p = () -> rand(Beta(1.0, 1.0))
    dp = DP(1.0, p, 0.0, 1.0)
    return rej_rates(dp, dp, n_top, n_bottom, nPerms, nReps)
end



function tp_per_d(distances::Vector{Float64}, n_top::Int, n_bottom::Int, nPerms::Int, nReps::Int)
    # Estimates of true positive rates for different measures with given distance between them.
    # firsty finds two dps with given distance between them and then gets the TPR.

    r = [] # to store the rejection rates (TPR) for WW and HIPM
    d_found = [] # to store the distances for which we found the two dps
    for d in distances
        # Firstly find two betas with wasserstein distance between them close to d
        p_a, p_b, q_a, q_b = find_betas(d)
        if p_a < 0.0
            println("No beta distributions found for d = $d")
            continue
        else
            # Then compute the rejection rates
            println("Found beta distributions for d = $d")
            p = () -> rand(Beta(p_a, p_b))
            q = () -> rand(Beta(q_a, q_b))
            dp1 = DP(1.0, p, 0.0, 1.0)
            dp2 = DP(1.0, q, 0.0, 1.0)
            r_ww, r_lip = rej_rates(dp1, dp2, n_top, n_bottom, nPerms, nReps)
            push!(r, (r_ww, r_lip))
            push!(d_found, d)
        end
    end
    return d_found,r
end

