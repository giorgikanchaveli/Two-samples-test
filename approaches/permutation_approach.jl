include("../distributions.jl")

# using permutation approach we estimate the threshold for the distance between two empirical pm. 
# Given two empirical measures, we estimate the threshold by permuting them and computing distances.
using Plots
using ExtractMacro
using Random, Statistics




# IID

function sample_distances(dist::Function, p::emp_pm, q::emp_pm, perm::param_perm, seed = 1234)
    # samples distances between empirical measures from permutation approach
    # this is for i.i.d case
    @extract perm : n_shuffles 
    
    atoms = vcat(p.atoms, q.atoms) # all atoms from which we distribute them to two measures
    n_p = length(p.atoms)
    d_shuffled = zeros(n_shuffles) # distances recorded for each shuffled measures
    Random.seed!(seed)
    seeds = [rand(1:1000000) for i in 1:n_shuffles] # we need it to ensure that we generate same   
                # empirical measures for exchangeable and i.i.d case when m = 1
    for i in 1:n_shuffles
        if i % 100 == 0
            println("iteration (iid, permutation): $i")
        end
        # shuffle measures and record distances
        Random.seed!(seeds[i]) # we need it because dlip is changing the seed. 
        indices = randperm(n_p + length(q.atoms)) # indices for shuffled measures
       
        #shuffled_atoms = shuffle(atoms)
        atoms_p, atoms_q = atoms[indices[1:n_p]], atoms[indices[n_p+1:end]]
        d_shuffled[i] = dist(atoms_p, atoms_q)
    end
    return d_shuffled
end

function sample_distances(dist::Function, p::PM, q::PM, n::Int, perm::param_perm)
    # firstly we generate empirical measures from p and q and then we sample distances between them
    return sample_distances(dist, generate_emp(p, n), generate_emp(q, n), perm)
end

function perm_thresholds(dist::Function, p_emp::emp_pm, q_emp::emp_pm, perm::param_perm, θ::Vector{Float64}, seed = 1234)
    # returns the quantile of the distances between empirical measures from permutation approach
    test_stat_samples = sqrt(length(p_emp.atoms)/2) * sort(sample_distances(dist, p_emp, q_emp, perm, seed))
    return quantile(test_stat_samples, 1 .- θ, sorted = true)
end


# Exch


function sample_distances(dist::Function, p::emp_ppm, q::emp_ppm, params::param_perm, seed = 1234)
    # using permutation it estimates the quantile
    # it is for exchangeable case
    n_shuffles = params.n_shuffles

    d_shuffled = zeros(n_shuffles) # distances recorded for each shuffled measures
    Random.seed!(seed)
    seeds = [rand(1:1000000) for i in 1:n_shuffles] # we need it to ensure that we generate same   
                # empirical measures for exchangeable and i.i.d case when m = 1
    total_atoms = vcat(p.atoms, q.atoms)
    for i in 1:n_shuffles
        if i % 10 == 0
            println("iteration (exch, permutation): $i")
        end
        # permute data and record distances
        Random.seed!(seeds[i]) # we need it because dlip is changing the seed.
        indices = randperm(p.n + q.n)
        
        #indices = shuffle(1:p.n + q.n)
        p_shuffled = emp_ppm(total_atoms[indices[1:p.n],:], p.n, p.m , p.a, p.b)
        q_shuffled = emp_ppm(total_atoms[indices[p.n+1:end],:], q.n, q.m, q.a, q.b)
        d_shuffled[i] = dist(p_shuffled, q_shuffled) 
    end
    return d_shuffled
end

function sample_distances(dist::Function, p::PPM, q::PPM, n::Int, m::Int, params::param_perm)
    # we first generate empirical measure and then we sample distances between them using permutation approach
    return sample_distances(dist, generate_emp(p, n, m), generate_emp(q, n, m), params)
end



function perm_thresholds(dist::Function, emp_p::emp_ppm, emp_q::emp_ppm, perm::param_perm, θ::Vector{Float64}, seed = 1234)
    # returns the quantile of the distances between empirical measures from permutation approach
    d_shuffled = sqrt(emp_p.n/2) * sort(sample_distances(dist, emp_p, emp_q, perm, seed))
    return quantile(d_shuffled, 1 .- θ, sorted = true)
end

