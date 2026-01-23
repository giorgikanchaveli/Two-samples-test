#  In this file, we define structures and a way to generate Hierarchical sample for several types of laws of random probability measures.
#  In addition, we define functions to generate probability (actually parameters for it) measures directly from given laws of RPMs.


# Notes:
#        1) We are sorting rows after generating to speed up permutation test using WoW. 
#        2) generate_prob measures function generates one parameter which completely characterizes the probability measure.   
#           In the case of DM method, instead of doing the test on hierarchical samples, we do the test on quantiles obtained
#           from the vectors of probability measures, because this way their method is much faster. 


include("structures.jl")

using Distributions, Random, StatsBase


"""
    tnormal_normal

struct for Q, law of RPM, where random probability measure is Gaussian(δ, 1) and δ has truncated normal distribution on
interval [a,b] with standard deviation σ and mean μ.     

# Arguments:
    μ::Float64  :  mean of truncated normal distribution from which we generate mean of inner normal distribution
    σ::Float64  :  standard deviation of truncated normal distribution from which we generate mean of inner normal distribution
    a::Float64  :  left end of truncation interval
    b::Float64  :  right end of truncation interval 

"""
struct tnormal_normal<:LawRPM
    μ::Float64 
    σ::Float64 
    a::Float64 
    b::Float64 
    function tnormal_normal(μ, σ, a, b)
        
        a <= b || throw(ArgumentError("a must be less than b (got a=$a, b=$b)"))


        σ > 0 || throw(ArgumentError("Standard deviation σ must be positive (got σ=$σ)"))
        return new(Float64(μ), Float64(σ), Float64(a), Float64(b))
    end
end



"""
    DP

struct for the Dirichlet Process DP(α, P_0).

# Arguments:
    α::Float64         :  concentration parameter.
    p_0::Distribution  :  Base distribution.

"""
struct DP<:LawRPM 
    α::Float64
    p_0::Distribution 
    function DP(α, p_0::Distribution)
        α > 0 || throw(ArgumentError("Concentration parameter α must be positive (got α=$α)"))
        return new(Float64(α), p_0)
    end
end


"""
    discr_normal
    
struct for Q, law of RPM, where random probability measure is Gaussian(δ, 1) and δ has discrete distribution 
on some atoms in with specified weights.

# Arguments:
    atoms::Vector{Float64}    :  support points of discrete distribution.
    weights::Vector{Float64}  :  weights associated to each atom.
"""
struct discr_normal<:LawRPM
    atoms::Vector{Float64}
    weights::Vector{Float64}
    function discr_normal(atoms::Vector{Float64}, weights::Vector{Float64})
        length(atoms) == length(weights) || throw(ArgumentError("Length of atoms and weights must be equal. Got $(length(atoms)) and $(length(weights))."))
        sum_weights = sum(weights)
        abs(sum_weights - 1.0) < 1e-8 || throw(ArgumentError("Weights must sum to 1. Got sum = $sum_weights."))
        return new(atoms, weights)
    end
end


"""
    mixture
    
struct for Q, law of RPM, defined as Q = λQ_1 + (1 - λ)Q_2 where Q_1,Q_2 are another laws of RPM.

# Arguments:
    lawrpm1::LawRPM  :  Law of RPM
    lawrpm2::LawRPM  :  Law of RPM
    λ::Float64       :  mixing parameter

"""
struct mixture{Q1<:LawRPM, Q2<:LawRPM} <: LawRPM
    lawrpm1::Q1
    lawrpm2::Q2
    λ::Float64
    function mixture(lawrpm1::Q1, lawrpm2::Q2, λ::Float64) where {Q1<:LawRPM, Q2<:LawRPM}
        (0.0 <= λ <= 1.0) || throw(ArgumentError("Mixing parameter λ must be in [0,1]. Got λ = $λ."))
        return new(lawrpm1, lawrpm2, Float64(λ))
    end
end




# methods to generate hierarhical samples

"""
    generate_hiersample

Function to generate HierSample using given law of RPM.
# Arguments:
    lawrpm::LawRPM
    n::Int  :  number of rows in a hierarchical sample.
    m::Int  :  number of columns in a hierarchical sample.
"""
function generate_hiersample(lawrpm::LawRPM, n::Int, m::Int)
    atoms = Matrix{Float64}(undef, n, m)
    for i in 1:n
        atoms[i, :] = sort(sample_exch_seq(lawrpm, m)) # generate row of exchangeable sequences from law of RPM
    end
    a = @views minimum(atoms[:,1]) # left end point of interval containing hierarchical sample
    b = @views maximum(atoms[:,end]) # right end point of interval containing hierarchical
    return HierSample(atoms, a, b)
end


"""
    sample_exch_seq
Function to generate m samples from Dirichlet process DP(α, P_0). Start with a sample from base distribution P_0. Then, 
At each step i, with probability α / (α + i - 1) we generate a new sample 
from base distribution P_0, otherwise we choose uniformly from the previous samples.

# Arguments:
    lawrpm::DP  :  Dirichlet process DP(α, P_0)
    m::Int     :  number of samples to generate.
"""
 function sample_exch_seq(lawrpm::DP, m::Int)
    α = lawrpm.α
    p_0 = lawrpm.p_0
    samples = Vector{Float64}(undef, m) # to store samples from DP
    samples[1] = rand(p_0) # first sample from base distribution
    for i in 2:m
        if rand() <= α / (α + i - 1)
            samples[i] = rand(p_0) # new sample from base distribution
        else
            index = rand(1:(i-1)) 
            samples[i] = samples[index] # choose uniformly from the previous samples
        end
    end
    return samples
end

"""
    sample_exch_seq
Function to generate m samples from tnormal_normal law of RPM. Firstly, δ is generated from truncated normal distribution on [a,b]
with standard deviation σ and mean μ, and then i.i.d observations are generated from Gaussian(δ, 1).

# Arguments:
    lawrpm::tnormal_normal
    m::Int  :  number of samples to generate.
"""

function sample_exch_seq(lawrpm::tnormal_normal, m::Int)
    truncated_normal = truncated(Normal(lawrpm.μ, lawrpm.σ), lawrpm.a, lawrpm.b) 
    δ = rand(truncated_normal) # generate mean for latent Gaussian(δ, 1).
    return rand(Normal(δ, 1.0), m) # generate obseravtions from Gaussian(δ, 1).
end

"""
    sample_exch_seq
Function to generate m samples from discr_normal law of RPM. Firstly, δ is generated from discrete measure and 
then i.i.d observations are generated from Gaussian(δ, 1).
# Arguments:
    lawrpm::discr_normal
    m::Int  :  number of samples to generate.
"""
function sample_exch_seq(lawrpm::discr_normal, m::Int)
    δ = sample(lawrpm.atoms, Weights(lawrpm.weights), 1)[1]  # generate mean for latent Gaussian with parameters (δ, 1).
    return rand(Normal(δ, 1.0), m) # generate obseravtions from Gaussian(δ, 1).
end

"""
    sample_exch_seq
Function to generate m samples from mixture of two laws of RPM. Each latent probability measure is generated from either lawrpm1
or lawrpm2 with probabilities λ and 1 - λ respectively. Then, i.i.d observations are generated from it. 

# Arguments:
    lawrpm::mixture  :  mixture of two laws of RPM.
    m::Int           :  number of samples to generate.

"""

function sample_exch_seq(lawrpm::mixture, m::Int)
    λ = lawrpm.λ
    if rand() <= λ
        return sample_exch_seq(lawrpm.lawrpm1, m) # generate samples from lawrpm1
    else
        return sample_exch_seq(lawrpm.lawrpm2, m) # generate samples from lawrpm2
    end
end













# """
#     generate_hiersample

# Function to generate HierSample using tnormal_normal as a law of RPM. For each n row in a resulting hierarchical sample,
# δ is generated from truncated normal distribution on [a,b] with standard deviation σ and mean μ, and 
# then m observations are generated from Gaussian(δ, 1).

# # Arguments:
#     lawrpm::tnormal_normal
#     n::Int  :  number of rows in a hierarchical sample.
#     m::Int  :  number of columns in a hierarchical sample.
# """
# function generate_hiersample(lawrpm::tnormal_normal, n::Int, m::Int)
#     atoms = zeros(n,m)
#     for i in 1:n
#         truncated_normal = truncated(Normal(lawrpm.μ, lawrpm.σ), lawrpm.a, lawrpm.b) 
#         δ = rand(truncated_normal) # generate mean for latent Gaussian(δ, 1).

#         atoms[i,:] = sort(rand(Normal(δ, 1.0), m)) # generate obseravtions from Gaussian(δ, 1).
#     end
#     a = @views minimum(atoms[:,1]) # left end point of interval containing hierarchical sample
#     b = @views maximum(atoms[:,end]) # right end point of interval containing hierarchical sample
#     return HierSample(atoms, a, b)
# end



# """
#     generate_hiersample

# Function to generate HierSample using mixture of two laws of RPM. Each latent probability measure is generated from either lawrpm1
# or lawrpm2 with probabilities λ and 1 - λ respectively.

# # Arguments:
#     lawrpm::mixture  :  mixture of two laws of RPM.
#     n::Int           :  number of rows in a hierarchical sample.
#     m::Int           :  number of columns in a hierarchical sample.
# """
# function generate_hiersample(lawrpm::mixture, n::Int, m::Int)
#     atoms = zeros(n,m) # to store hierarchical sample.
#     λ = lawrpm.λ
#     for i in 1:n
#         if rand() <= λ
#             atoms[i,:] = generate_hiersample(lawrpm.lawrpm1, 1, m).atoms[1,:] # generate one row from lawrpm1
#         else
#             atoms[i,:] = generate_hiersample(lawrpm.lawrpm2, 1, m).atoms[1,:] # generate one row from lawrpm2
#         end
#     end
#     a = @views minimum(atoms[:,1]) # left end point of interval containing hierarchical sample
#     b = @views maximum(atoms[:,end]) # right end point of interval containing hierarchical sample
#     return HierSample(atoms, a, b)
# end


# """
#     generate_hiersample

# Function to generate HierSample using discr_normal as a law of RPM. For each n row in a resulting hierarchical sample,
# δ is generated from discrete measure and then m observations are generated from Gaussian(δ, 1).

# # Arguments:
#     lawrpm::discr_normal
#     n::Int  :  number of rows in a hierarchical sample.
#     m::Int  :  number of columns in a hierarchical sample.
# """

# function generate_hiersample(lawrpm::discr_normal, n::Int, m::Int)
#     atoms = zeros(n, m) # to store hierarchical sample.

#     δs = sample(lawrpm.atoms, Weights(lawrpm.weights), n)  # generate means for latent Gaussians with parameters (δ, 1).
#     for i in 1:n_top
#         atoms[i, :] = sort(rand(Normal(δs[i], 1.0), m)) # generate obseravtions from Gaussian(δ, 1).
#     end

#     a = @views minimum(atoms[:,1]) # left end point of interval containing hierarchical sample
#     b = @views maximum(atoms[:,end]) # right end point of interval containing hierarchical sample
#     return HierSample(atoms, a, b)
# end





# """
#     generate_prob_measures
# Function to generate n values for δ tnormal_normal law of RPM which completely characterizes the probability measure Gaussian(δ, 1).
# # Arguments:
#     lawppm::tnormal_normal
#     n::Int  :  number of probability measures to generate.
# """

# function generate_prob_measures(lawppm::tnormal_normal, n::Int)
#     truncated_normal = truncated(Normal(lawppm.μ, lawppm.σ), lawppm.a, lawppm.b)
#     return rand(truncated_normal, n)
# end

# """
#     generate_prob_measures
# Function to generate n values for δ from discr_normal that completely characterizes the probability measure Gaussian(δ, 1).
# As discr_normal is discrete measure, we sample n values from its atoms with respect to its weights (without replacement).

# # Arguments:
#     lawppm::discr_normal
#     n::Int  :  number of probability measures to generate.
# """
# function generate_prob_measures(lawppm::discr_normal, n::Int)
#     return sample(lawppm.atoms, Weights(lawppm.weights), n)
# end

# """
#     generate_prob_measures

# Function to generate n values for δ from mixture of two laws of RPM that completely characterizes the probability measure Gaussian(δ, 1).
# Each δ is generated from either lawppm1 or lawppm2 with probabilities λ and 1 - λ respectively. 

# # Arguments:
#     lawppm::mixture
#     n::Int  :  number of probability measures to generate.
# """
# function generate_prob_measures(lawppm::mixture, n::Int)
#     δs = zeros(n)
#     λ = lawppm.λ # mixing parameter
#     for i in 1:n
#         if rand() <= λ
#             δs[i] = generate_prob_measures(lawppm.ppm1, 1)[1]
#         else
#             δs[i] = generate_prob_measures(lawppm.ppm2, 1)[1]
#         end
#     end
#     return δs
# end

# function issorted(m::Matrix{Float64})
#     # Checks if each row of matrix m is sorted.
#     flag = true
#     for i in m.size[1]
#         row = m[i,:]
#         for i in 1:(length(row)-1)
#             if row[i] > row[i + 1]
#                 flag = false
#                 return flag
#             end
#         end
#     end
#     return flag
# end


# function probability(baseMeasure::String)
#     # function to generate observation either from uniform(-1/2,1/2) or from splitting measure
#     if baseMeasure == "same" # Uniform(-1/2,1/2)
#         return rand() - 0.5
#     elseif baseMeasure == "splitting"  # sample either close to -1 or close to 1
#         atom = rand()
#         mixture = rand((0,1))
#         return mixture * ( -1. + 0.25 * atom ) + (1 - mixture) * (0.75 + 0.25 * atom)
#     end
# end


# structures for laws of RPMs.

# These structures encapsulate all the parameters needed to generated hierarchical samples from given laws of RPM.


# struct discrrpm<:LawRPM
#     # discrete random probability measure
#     # sample space is [a,b]
#     atoms::Matrix{Float64} # n_top x n_bottom matrix of atoms
#     weights::Vector{Float64} # vector of weights for each probability measure
#     n_top::Int # number of probability measures from which observations are generated
#     n_bottom::Int # number of observations from each probability measure
#     a::Float64 # interval [a,b] from which atoms are drawn
#     b::Float64 # interval [a,b] from which atoms are drawn

#     function discrrpm(n_top::Int, n_bottom::Int, a::Float64, b::Float64)
#         # given n_top and n_bottom constructs discrete random probability measure with atoms in [a,b] and weights
#         atoms = rand(Uniform(a,b),n_top,n_bottom)
#         weights = rand(n_top)
#         weights = weights ./ sum(weights)
#         new(atoms,weights,n_top,n_bottom,a,b)
#     end

# end


# struct discr_tnormal<:LawRPM
#     # discrete measure over n_1 truncated gaussian distributions on [a,b]
#     # sample space is [a,b]

#     n_1::Int # number of normal distributions
#     μ::Vector{Float64} # mean of each normal distribution
#     σ::Vector{Float64} # standard deviation each of normal distribution
#     a::Float64 # interval [a,b] from which atoms are drawn
#     b::Float64 # interval [a,b] from which atoms are drawn
# end





# struct discrete_normal<:LawRPM
#     # law of RPM is a discrete measure over normal distributions with varying mean but variance equal to 1.

#     μs::Vector{Float64} # Vector of means of normal distributions
#     weights::Vector{Float64} # Vector of weights associated to each normal distribution
# end




# function generate_hiersample(ppm::discrrpm, n_top::Int, n_bottom::Int)
#     # given discrete random probability measure, generates empirical measure struct
#     # n is the number of random probability measures we want to sample
#     # m is the number of atoms from each random probability measure
#     a,b = ppm.a, ppm.b
#     atoms = zeros(n_top,n_bottom)
    
#     r = Categorical(ppm.weights) # used to sample from random probability measure
#     for i in 1:n_top
#         prob = rand(r) # probability measure sampled from discrete rpm
#         atoms[i,:] = rand(ppm.atoms[prob,:], n_bottom) 
#     end
#     return HierSample(atoms, n_top, n_bottom, a, b)
# end


# function generate_hiersample(ppm::discr_tnormal, n_top::Int, n_bottom::Int)
#     # given discrete random probability measure over truncated normal distributions, generates empirical measure struct
#     # which is used for estimating law of rpm.
#     # n is the number of random probability measures we want to sample
#     # m is the number of atoms from each random probability measure
#     a,b = ppm.a, ppm.b
#     atoms = zeros(n_top,n_bottom)
#     for i in 1:n_top
#         r = rand(1:ppm.n_1) # indexes the probability measure from which we sample i.i.d observations
#         truncated_normal = truncated(Normal(ppm.μ[r], ppm.σ[r]), a, b)
#         atoms[i,:] = rand(truncated_normal, n_bottom)
#     end
#     return HierSample(atoms, a, b)
# end



# function generate_hiersample(pms::Vector{Normal}, n_top::Int, n_bottom::Int)
#     # given vector of n_top normal distributions, we generate n_bottom observations from each of them.
#     # length sequence of observations. 
#     @assert length(pms) == n_top "n and length of vector of probability measures are not equal"
#     atoms = zeros(n_top,n_bottom)
#     for i in 1:n_top
#         atoms[i,:] = rand(pms[i], n_bottom)
#     end
#     a = minimum(atoms) # left end of an interaval where observations take values
#     b = maximum(atoms) # right end of an interaval where observations take values
#     return HierSample(atoms, a, b)
# end

# function generate_hiersample(pms::Vector{Any}, n_top::Int, n_bottom::Int)
#     # given vector of n_top any probability measures, we generate n_bottom observations from each of them.

#     # Each element in pms vector must be of type distributino ? 

#     @assert length(pms) == n_top "n and length of vector of probability measures are not equal"
#     atoms = zeros(n_top,n_bottom)
#     for i in 1:n_top
#         atoms[i,:] = rand(pms[i], n_bottom)
#     end
#     a = minimum(atoms) # left end of an interaval where observations take values
#     b = maximum(atoms) # right end of an interaval where observations take values
#     return HierSample(atoms, a, b)
# end



# Methods to generate probability measures as samples. 

# They are useful, because sometimes we might generate firstly probability measure and then obseravtions from them.




# function generate_prob_measures(ppm::tnormal_normal, n_top::Int)
#     # given law of random probability measure which is truncated normal, generate
#     # n_top normal distributions which is Normal(μ,1) and save it into a vector.
#     pms = Vector{Normal}(undef, n_top)
#     t_normal = truncated(Normal(ppm.μ, ppm.σ), ppm.a, ppm.b)
#     for i in 1:n_top
#         μ = rand(t_normal)
#         pms[i] = Normal(μ, 1.0) # i-th probability measure
#     end    
#     return pms
# end

# function generate_prob_measures(ppm::normal_normal, n_top::Int)
#     # given law of random probability measure which is normal, generate
#     # n_top normal distributions with variance 1 and save it into a vector.
#     pms = Vector{Normal}(undef, n_top)
#     normal_distr = Normal(ppm.μ, ppm.σ)
#     for i in 1:n_top
#         μ = rand(normal_distr)
#         pms[i] = Normal(μ, 1.0) # i-th probability measure
#     end    
#     return pms
# end


# function generate_prob_measures(ppm::uniform_normal, n_top::Int)
#     # given law of random probability measure which is uniform on [-1,1], generate
#     # n_top normal distributions with variance 1.0 and save it into a vector.
#     pms = Vector{Normal}(undef, n_top)
#     unif = Uniform(ppm.a, ppm.b)
#     for i in 1:n_top
#         μ = rand(unif)
#         pms[i] = Normal(μ, 1.0) # i-th probability measure
#     end    
#     return pms
# end


# function generate_prob_measures(ppm::discrete_normal)
#     pms = Vector{Normal}(undef, n_top)
#     # we sample mean from ppm.means with respect to ppm.weights
#     index_generator = Categorical(ppm.weights)
#     for i in 1:n_top
#         index = rand(index_generator) # generate index for mean from ppm.means using probabilities = ppm.weights
#         pms[i] = Normal(ppm.μs[index], 1.0) # i-th probability measure
#     end    
#     return pms
# end



# function generate_prob_measures(ppm::mixture_ppm, n_top::Int)
#     # given law of random probability measure which mixture of two distributions, generate
#     # n_top probability measures from it
#     pms = Vector{Any}(undef, n_top) 
#     for i in 1:n_top
#         if rand() <= ppm.λ
#             pms[i] = generate_prob_measures(ppm.ppm1, 1)[1]
#         else
#             pms[i] = generate_prob_measures(ppm.ppm2, 1)[1]
#         end
#     end
#     return pms
# end







# function generate_hiersample(ppm::LawRPM, n_top::Int, n_bottom::Int)
#     # given law of random probability measure, instead of directly generating hierarchical samples, we firstly
#     # generate probability meassures and then observations from them.
#     pms = generate_prob_measures(ppm, n_top)
#     return generate_hiersample(pms, n_top, n_bottom)
# end

# struct normal_tnormal<:LawRPM
#     # random probability measure is truncated gaussian distributions on [a,b] with mean generated from normal(μ, σ)
#     # sample space is [a,b]

#     μ::Float64 # mean of normal distribution from which we generate mean of inner normal distribution
#     σ::Float64 # standard deviation of normal distribution from which we generate mean of inner normal distribution
#     a::Float64 # interval [a,b] from which atoms are drawn
#     b::Float64 # interval [a,b] from which atoms are drawn
# end



# struct normal_normal<:LawRPM 
#     # random probability measure is gaussian distributions with variance equal to 1 and 
#     # mean generated from normal(μ, σ) 
    
#     # Note that sample space for exhangeable sequences is R

#     μ::Float64 # mean of normal distribution from which we generate mean of inner normal distribution
#     σ::Float64 # standard deviation of normal distribution from which we generate mean of inner normal distribution
# end



# struct uniform_normal<:LawRPM
#     # random probability measure is normal distribution with variance 1 and with mean generated from uniform(a,b)
    
#     # Note that sample space for exhangeable sequences is R
#     a::Float64 
#     b::Float64 
# end








# function generate_hiersample(ppm::normal_tnormal, n_top::Int, n_bottom::Int)
#     # given random probability measure which takes values as trunacted normal distributions where mean is random variable from normal(μ,σ), 
#     # we generate hieraarchical measure
#     # which is used for estimating law of rpm.

#     # n is the number of random probability measures we want to sample
#     # m is the number of atoms from each random probability measure
#     a,b = ppm.a, ppm.b
#     atoms = zeros(n_top,n_bottom)
#     for i in 1:n_top
#         μ_inner = rand(Normal(ppm.μ, ppm.σ))
#         truncated_normal = truncated(Normal(μ_inner, ppm.σ), a, b)
#         atoms[i,:] = sort(rand(truncated_normal, n_bottom))
#     end
#     return HierSample(atoms, a, b)
# end
