include("structures.jl")
using Distributions, Random


function probability(baseMeasure::String)
    # function to generate observation either from uniform(-1/2,1/2) or from splitting measure
    if baseMeasure == "same" # Uniform(-1/2,1/2)
        return rand() - 0.5
    elseif baseMeasure == "splitting"  # sample either close to -1 or close to 1
        atom = rand()
        mixture = rand((0,1))
        return mixture * ( -1. + 0.25 * atom ) + (1 - mixture) * (0.75 + 0.25 * atom)
    end

end


# Exchangeable
struct discrrpm<:PPM
    # discrete random probability measure
    # sample space is [a,b]
    atoms::Matrix{Float64} # n_top x n_bottom matrix of atoms
    weights::Vector{Float64} # vector of weights for each probability measure
    n_top::Int # number of probability measures from which observations are generated
    n_bottom::Int # number of observations from each probability measure
    a::Float64 # interval [a,b] from which atoms are drawn
    b::Float64 # interval [a,b] from which atoms are drawn

    function discrrpm(n_top::Int, n_bottom::Int, a::Float64, b::Float64)
        # given n_top and n_bottom constructs discrete random probability measure with atoms in [a,b] and weights
        atoms = rand(Uniform(a,b),n_top,n_bottom)
        weights = rand(n_top)
        weights = weights ./ sum(weights)
        new(atoms,weights,n_top,n_bottom,a,b)
    end

end


struct discr_normal<:PPM
    # discrete measure over n_1 truncated gaussian distributions on [a,b]
    # sample space is [a,b]

    n_1::Int # number of normal distributions
    μ::Vector{Float64} # mean of each normal distribution
    σ::Vector{Float64} # standard deviation each of normal distribution
    a::Float64 # interval [a,b] from which atoms are drawn
    b::Float64 # interval [a,b] from which atoms are drawn
end


struct normal_normal<:PPM
    # random probability measure is truncated gaussian distributions on [a,b] with mean generated from normal(μ, σ)
    # sample space is [a,b]

    μ::Float64 # mean of normal distribution from which we generate mean of inner normal distribution
    σ::Float64 # standard deviation of normal distribution from which we generate mean of inner normal distribution
    a::Float64 # interval [a,b] from which atoms are drawn
    b::Float64 # interval [a,b] from which atoms are drawn
end


struct tnormal_normal<:PPM
    # random probability measure is gaussian distributions on R with mean generated from truncated normal(μ, 0.5) on [-1,1]
    # sample space is R
    δ::Float64 # mean of normal distribution from which we generate mean of inner normal distribution
end



struct DP<:PPM # Dirichlet process
    α::Float64
    p_0::Function # function generating observations from p_0
    a::Float64 
    b::Float64
    # [a,b] is the observation space
end

struct new_normal_normal<:PPM # previous normal normal should be changed to normal_tnormal
    # random probability measure is truncated gaussian distributions on [a,b] with mean generated from normal(μ, σ)
    # sample space is [a,b]

    δ::Float64 # mean of normal distribution from which we generate mean of inner normal distribution
    σ::Float64 # standard deviation of normal distribution from which we generate mean of inner normal distribution
end



struct normal_uniform<:PPM # previous normal normal should be changed to normal_tnormal
    # random probability measure is with mean generated from uniform(a,b)
    # sample space is [a,b]

    a::Float64 
    b::Float64 
end







function dirichlet_process_without_weight(n, α, p_0)
    # auxiliary function
    # Given function p_0 that returns sample from chosen probability measure P_0
    # we generate a n-sample from a Dirichlet process with parameter α and p_0
    if n == 0
        return []
    elseif n == 1 # It's not needed in general but useful to have same seed for iid and exch. case
        prev = dirichlet_process_without_weight(n-1,α,p_0)
        return push!(prev, p_0())
    else 
        prev = dirichlet_process_without_weight(n-1,α,p_0)
        if rand() <= α /(α +n-1) # sample from P_0
            return push!(prev, p_0())
        else # sample from the already given observations
            index = rand(1:(n-1))
            return push!(prev, prev[index])
        end
    end  
end

function generate_emp(ppm::DP, n_top::Int, n_bottom::Int)
    # given random probability measure from Dirichlet process, generates empirical measure struct
    # which is used for estimating law of rpm.
    # n is the number of random probability measures we want to sample
    # m is the number of atoms from each random probability measure
    atoms = zeros(n_top,n_bottom)
    for i in 1:n_top
        atoms[i,:] = dirichlet_process_without_weight(n_bottom, ppm.α, ppm.p_0)
    end
    return emp_ppm(atoms, n_top, n_bottom, ppm.a, ppm.b)
end

function generate_emp(ppm::discrrpm, n_top::Int, n_bottom::Int)
    # given discrete random probability measure, generates empirical measure struct
    # n is the number of random probability measures we want to sample
    # m is the number of atoms from each random probability measure
    a,b = ppm.a, ppm.b
    atoms = zeros(n_top,n_bottom)
    
    r = Categorical(ppm.weights) # used to sample from random probability measure
    for i in 1:n_top
        prob = rand(r) # probability measure sampled from discrete rpm
        atoms[i,:] = rand(ppm.atoms[prob,:], n_bottom) 
    end
    return emp_ppm(atoms, n_top, n_bottom, a, b)
end


function generate_emp(ppm::discr_normal, n_top::Int, n_bottom::Int)
    # given discrete random probability measure over truncated normal distributions, generates empirical measure struct
    # which is used for estimating law of rpm.
    # n is the number of random probability measures we want to sample
    # m is the number of atoms from each random probability measure
    a,b = ppm.a, ppm.b
    atoms = zeros(n_top,n_bottom)
    for i in 1:n_top
        r = rand(1:ppm.n_1) # indexes the probability measure from which we sample i.i.d observations
        truncated_normal = truncated(Normal(ppm.μ[r], ppm.σ[r]), a, b)
        atoms[i,:] = rand(truncated_normal, n_bottom)
    end
    return emp_ppm(atoms, n_top, n_bottom, a, b)
end

function generate_emp(ppm::normal_normal, n_top::Int, n_bottom::Int)
    # given random probability measure which takes values as normal distributions where mean is random variable from normal(μ,σ), we generate hieraarchical measure
    # which is used for estimating law of rpm.

    # n is the number of random probability measures we want to sample
    # m is the number of atoms from each random probability measure
    a,b = ppm.a, ppm.b
    atoms = zeros(n_top,n_bottom)
    for i in 1:n_top
        μ_inner = rand(Normal(ppm.μ, ppm.σ))
        truncated_normal = truncated(Normal(μ_inner, ppm.σ), a, b)
        atoms[i,:] = rand(truncated_normal, n_bottom)
    end
    return emp_ppm(atoms, n_top, n_bottom, a, b)
end

function generate_emp(pms::Vector{Normal}, n_top::Int, n_bottom::Int)
    # given vector of n_top probability measures. From each of them generate the n_bottom
    # length sequence of observations. 
    @assert length(pms) == n_top "n and length of vector of probability measures are not equal"
    atoms = zeros(n_top,n_bottom)
    for i in 1:n_top
        atoms[i,:] = rand(pms[i], n_bottom)
    end
    a = minimum(atoms) # left end of an interaval where observations take values
    b = maximum(atoms) # right end of an interaval where observations take values
    return emp_ppm(atoms, n_top, n_bottom, a, b)
end

function generate_prob_measures(ppm::tnormal_normal, n_top::Int)
    # given law of random probability measure which is truncated normal, generate
    # n_top normal distributions and save it into a vector.
    pms = Vector{Normal}(undef, n_top)
    for i in 1:n_top
        t = truncated(Normal(ppm.δ, 0.5), -1.0, 1.0) # I think it should be sqrt(0.5)
        μ = rand(t)
        pms[i] = Normal(μ, 1.0) # i-th probability measure
    end    
    return pms
end

function generate_prob_measures(ppm::new_normal_normal, n_top::Int)
    # given law of random probability measure which is truncated normal, generate
    # n_top normal distributions and save it into a vector.
    pms = Vector{Normal}(undef, n_top)
    for i in 1:n_top
        t = Normal(ppm.δ, ppm.σ) 
        μ = rand(t)
        pms[i] = Normal(μ, 1.0) # i-th probability measure
    end    
    return pms
end


function generate_prob_measures(ppm::normal_uniform, n_top::Int)
    # given law of random probability measure which is uniform on [-1,1], generate
    # n_top normal distributions and save it into a vector.
    pms = Vector{Normal}(undef, n_top)
    for i in 1:n_top
        t = Uniform(ppm.a, ppm.b)
        μ = rand(t)
        pms[i] = Normal(μ, 1.0) # i-th probability measure
    end    
    return pms
end


function generate_emp(ppm::tnormal_normal, n_top::Int, n_bottom::Int)
    # given law of random probability measure which is truncated normal, generate
    # hierarchical sample
    pms = generate_prob_measures(ppm, n_top)
    return generate_emp(pms, n_top, n_bottom)
end

function generate_emp(ppm::new_normal_normal, n_top::Int, n_bottom::Int)
    # given law of random probability measure which is normal, generate
    # hierarchical sample
    pms = generate_prob_measures(ppm, n_top)
    return generate_emp(pms, n_top, n_bottom)
end

function generate_emp(ppm::normal_uniform, n_top::Int, n_bottom::Int)
    # given law of random probability measure which is uniform on [-1,1], generate
    # hierarchical sample
    pms = generate_prob_measures(ppm, n_top)
    return generate_emp(pms, n_top, n_bottom)
end