#  This file contains code to generate Hierarchical samples from several types of laws of random probability measures.

#  In addition, we define functions to generate probability measures directly from given laws of RPMs. This is achieved by generating parameters that charaterize those measures. Such methods are useful for Frechet test as we use at the level of probability measures (by computing quantiles, see functions decision_dm in methods.jl). This way their methods is much faster.


# Important Notes:
#        1) In the generation of hierarchical samples, we sort rows containing the atoms to speed up permutation test using WoW. 


include("structures.jl")

using Distributions, Random, StatsBase


# Specific structs for laws of RPMs

"""
    beta_beta_A

Struct for law of RPM where P ∼ Beta(μ, 1-μ) and μ∼Beta(a, b). 
(Unscaled inner Beta;
compare `beta_beta_B`, which uses Beta(2μ, 2(1-μ)).)

# Arguments:
    a::Float64  :  Parameter for outer beta distribution.
    b::Float64  :  Parameter for outer beta distribution.

"""
struct beta_beta_A<:LawRPM
    a::Float64
    b::Float64
    function beta_beta_A(a::Float64, b::Float64)
        a > 0 || throw(ArgumentError("a must be larger than 0"))
        b > 0 || throw(ArgumentError("b must be larger than 0"))
        return new(a, b)
    end
end


"""
    beta_beta_B

Struct for law of RPM where P ∼ Beta(2μ, 2*(1-μ)) and μ∼Beta(a, b).

# Arguments:
    a::Float64  :  Parameter for outer beta distribution.
    b::Float64  :  Parameter for outer beta distribution.

"""
struct beta_beta_B<:LawRPM
    a::Float64
    b::Float64
    function beta_beta_B(a::Float64, b::Float64)
        a > 0 || throw(ArgumentError("a must be larger than 0"))
        b > 0 || throw(ArgumentError("b must be larger than 0"))
        return new(a, b)
    end
end



"""
    normal_normal_A

Struct for law of RPM where P ∼ Normal(δ, 1) and δ∼Normal(μ, σ^2)). (Fixed unit inner variance;
compare `normal_normal_B`, which lets the inner variance vary via σ_inner.)

# Arguments:
    μ::Float64  :  mean for outer normal distribution.
    σ::Float64  :  standard deviation for outer normal distribution.

"""
struct normal_normal_A<:LawRPM
    μ::Float64
    σ::Float64
    function normal_normal_A(μ::Float64, σ::Float64)
        σ > 0 || throw(ArgumentError("σ must be larger than 0"))
        return new(μ, σ)
    end
end

"""
    normal_normal_B

Struct for law of RPM where P ∼ Normal(δ, σ_inner^2) and δ∼Normal(μ, σ_outer^2).

# Arguments:
    μ::Float64  :  mean for outer normal distribution.
    σ_outer::Float64  :  standard deviation for outer normal distribution.
    σ_inner::Float64  :  standard deviation for innter normal distribution.

"""
struct normal_normal_B<:LawRPM
    μ::Float64
    σ_outer::Float64
    σ_inner::Float64
    function normal_normal_B(μ::Float64,σ_outer::Float64, σ_inner::Float64)
        σ_inner > 0 || throw(ArgumentError("σ_inner must be larger than 0"))
        σ_outer > 0 || throw(ArgumentError("σ_outer must be larger than 0"))
        return new(μ, σ_outer, σ_inner)
    end
end




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
    function tnormal_normal(μ::Float64, σ::Float64, a::Float64, b::Float64)
        
        a <= b || throw(ArgumentError("a must be less than b (got a=$a, b=$b)"))


        σ > 0 || throw(ArgumentError("Standard deviation σ must be positive (got σ=$σ)"))
        return new(μ, σ, a, b)
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
    function DP(α::Float64, p_0::Distribution)
        α > 0 || throw(ArgumentError("Concentration parameter α must be positive (got α=$α)"))
        return new(α, p_0)
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
    unif_spike

Struct for a law of RPM where ν = Unif[10, 10+L] and the latent measure is P_u = (1-a)δ_u + aδ_{spike}.

Concretely, given U ~ ν each observation equals U with probability (1-a) and spike with probability a.
Use spike=0.0 for the first population (Q^1) and spike=1.0 for the second population (Q^2).

# Arguments:
    L::Float64      :  length of the uniform support interval [10, 10+L]; must be positive.
    a::Float64      :  weight on the spike atom; must be in (0, 1).
    spike::Float64  :  location of the spike atom (e.g. 0.0 or 1.0).
"""
struct unif_spike <: LawRPM
    L::Float64
    a::Float64
    spike::Float64
    function unif_spike(L::Float64, a::Float64, spike::Float64)
        L > 0 || throw(ArgumentError("L must be positive (got L=$L)"))
        0 < a < 1 || throw(ArgumentError("a must be in (0,1) (got a=$a)"))
        return new(L, a, spike)
    end
end



"""
    discr_law

Struct for a discrete law of RPM. The latent probability measure is selected from a finite list of
distributions according to a weight vector, and sequence is generated from it.

# Arguments:
    weights::Vector{Float64}       :  probability of selecting each distribution; must be non-negative and sum to 1.
    distributions::Vector{<:UnivariateDistribution}  :  list of distributions to choose from (e.g. Normal(0,1), Uniform(0,1)).

"""
struct discr_law <: LawRPM
    weights::Vector{Float64}
    distributions::Vector{UnivariateDistribution}
    function discr_law(weights::Vector{Float64}, distributions::Vector{<:UnivariateDistribution})
        length(weights) == length(distributions) || throw(ArgumentError("Length of weights and distributions must be equal. Got $(length(weights)) and $(length(distributions))."))
        all(w >= 0 for w in weights) || throw(ArgumentError("All weights must be non-negative."))
        sum_weights = sum(weights)
        abs(sum_weights - 1.0) < 1e-8 || throw(ArgumentError("Weights must sum to 1. Got sum = $sum_weights."))
        return new(weights, Vector{UnivariateDistribution}(distributions))
    end
end


"""
    mixture

struct for Q, law of RPM, defined as Q = λQ_1 + (1 - λ)Q_2 where Q_1,Q_2 are another laws of RPM.

# Arguments:
    law_rpm_1::LawRPM  :  Law of RPM
    law_rpm_2::LawRPM  :  Law of RPM
    λ::Float64       :  mixing parameter

"""
struct mixture{T1<:LawRPM, T2<:LawRPM} <: LawRPM
    law_rpm_1::T1
    law_rpm_2::T2
    λ::Float64
    function mixture(law_rpm_1::T1, law_rpm_2::T2, λ::Float64) where {T1<:LawRPM, T2<:LawRPM}
        (0.0 <= λ <= 1.0) || throw(ArgumentError("Mixing parameter λ must be in [0,1]. Got λ = $λ."))
        return new{T1,T2}(law_rpm_1, law_rpm_2, λ)
    end
end




# Methods to generate hierarhical samples

"""
    generate_hiersample

Function to generate HierSample using given law of RPM.
# Arguments:
    law_rpm::LawRPM
    n::Int  :  number of rows in a hierarchical sample.
    m::Int  :  number of columns in a hierarchical sample.
"""
function generate_hiersample(law_rpm::LawRPM, n::Int, m::Int)
    atoms = Matrix{Float64}(undef, n, m)
    for i in 1:n
        atoms[i, :] = sort(sample_exch_seq(law_rpm, m)) # generate row of exchangeable sequences from law of RPM
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
    law_rpm::DP  :  Dirichlet process DP(α, P_0)
    m::Int     :  number of samples to generate.
"""
 function sample_exch_seq(law_rpm::DP, m::Int)
    α = law_rpm.α
    p_0 = law_rpm.p_0
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
Function to generate m samples from beta_beta_A law of RPM. Firstly, μ is generated from beta distribution and then i.i.d observations
are generated from beta(μ, 1-μ).

# Arguments:
    law_rpm::beta_beta_A
    m::Int  :  number of samples to generate.
"""

function sample_exch_seq(law_rpm::beta_beta_A, m::Int)

    μ = rand(Beta(law_rpm.a, law_rpm.b)) 
    return rand(Beta(μ, 1-μ), m) # generate obseravtions from Beta(μ, 1-μ).
end



"""
    sample_exch_seq
Function to generate m samples from beta_beta_B law of RPM. Firstly, μ is generated from beta distribution and then i.i.d observations
are generated from beta(2*μ, 2*(1-μ)).

# Arguments:
    law_rpm::beta_beta_B
    m::Int  :  number of samples to generate.
"""

function sample_exch_seq(law_rpm::beta_beta_B, m::Int)

    μ = rand(Beta(law_rpm.a, law_rpm.b)) 
    return rand(Beta(2*μ, 2*(1-μ)), m) # generate obseravtions from Beta(2*μ, 2*(1-μ)).
end




"""
    sample_exch_seq
Function to generate m samples from normal_normal_A law of RPM. Firstly, δ is generated from normal distribution
and then i.i.d observations are generated from Gaussian(δ, 1).

# Arguments:
    law_rpm::normal_normal_A
    m::Int  : number of samples to generate.
"""

function sample_exch_seq(law_rpm::normal_normal_A, m::Int)
    δ = rand(Normal(law_rpm.μ, law_rpm.σ)) 
    return rand(Normal(δ, 1.0), m) # generate obseravtions from Gaussian(δ, 1).
end


"""
    sample_exch_seq
Function to generate m samples from normal_normal_B law of RPM. Firstly, δ is generated from normal distribution
and then i.i.d observations are generated from Gaussian(δ, σ_inner^2).

# Arguments:
    law_rpm::normal_normal_B
    m::Int  : number of samples to generate.
"""

function sample_exch_seq(law_rpm::normal_normal_B, m::Int)
    δ = rand(Normal(law_rpm.μ, law_rpm.σ_outer)) 
    return rand(Normal(δ, law_rpm.σ_inner), m) # generate obseravtions from Gaussian(δ, σ_inner^2).
end



"""
    sample_exch_seq
Function to generate m samples from unif_spike law of RPM. First U is drawn from Unif[10, 10+L],
then each observation independently equals U with probability (1-a) and spike with probability a.

# Arguments:
    law_rpm::unif_spike
    m::Int  :  number of samples to generate.
"""
function sample_exch_seq(law_rpm::unif_spike, m::Int)
    U = rand(Uniform(10.0, 10.0 + law_rpm.L))
    return [rand() < law_rpm.a ? law_rpm.spike : U for _ in 1:m]
end







"""
    sample_exch_seq
Function to generate m samples from tnormal_normal law of RPM. Firstly, δ is generated from truncated normal distribution on [a,b]
with standard deviation σ and mean μ, and then i.i.d observations are generated from Gaussian(δ, 1).

# Arguments:
    law_rpm::tnormal_normal
    m::Int  :  number of samples to generate.
"""

function sample_exch_seq(law_rpm::tnormal_normal, m::Int)
    truncated_normal = truncated(Normal(law_rpm.μ, law_rpm.σ), law_rpm.a, law_rpm.b) 
    δ = rand(truncated_normal) # generate mean for latent Gaussian(δ, 1).
    return rand(Normal(δ, 1.0), m) # generate obseravtions from Gaussian(δ, 1).
end

"""
    sample_exch_seq
Function to generate m samples from discr_normal law of RPM. Firstly, δ is generated from discrete measure and 
then i.i.d observations are generated from Gaussian(δ, 1).
# Arguments:
    law_rpm::discr_normal
    m::Int  :  number of samples to generate.
"""
function sample_exch_seq(law_rpm::discr_normal, m::Int)
    δ = sample(law_rpm.atoms, Weights(law_rpm.weights), 1)[1]  # generate mean for latent Gaussian with parameters (δ, 1).
    return rand(Normal(δ, 1.0), m) # generate obseravtions from Gaussian(δ, 1).
end

"""
    sample_exch_seq
Function to generate m samples from mixture of two laws of RPM. Each latent probability measure is generated from either law_rpm_1
or law_rpm_2 with probabilities λ and 1 - λ respectively. Then, i.i.d observations are generated from it. 

# Arguments:
    law_rpm::mixture  :  mixture of two laws of RPM.
    m::Int           :  number of samples to generate.

"""

function sample_exch_seq(law_rpm::mixture, m::Int)
    λ = law_rpm.λ
    if rand() <= λ
        return sample_exch_seq(law_rpm.law_rpm_1, m) # generate samples from law_rpm_1
    else
        return sample_exch_seq(law_rpm.law_rpm_2, m) # generate samples from law_rpm_2
    end
end


"""
    sample_exch_seq
Function to generate m samples from discr_law law of RPM. A distribution is selected from the list
according to the weight vector, and then m i.i.d. observations are generated from it.

# Arguments:
    law_rpm::discr_law  :  discrete law of RPM.
    m::Int              :  number of samples to generate.
"""
function sample_exch_seq(law_rpm::discr_law, m::Int)
    idx = sample(1:length(law_rpm.distributions), Weights(law_rpm.weights))
    return rand(law_rpm.distributions[idx], m)
end


"""
    sample_exch_seq

Notifier for laws of RPM that don't have a specific sample_exch_seq method. Raises an informative error.

# Arguments:
    law_rpm::LawRPM
    m::Int  :  number of samples to generate.
"""
function sample_exch_seq(law_rpm::LawRPM, m::Int)
    throw(ArgumentError("sample_exch_seq is not defined for $(typeof(law_rpm))."))
end






"""
    generate_prob_measures
Function to generate n values of the δ parameter (not Distribution objects) from tnormal_normal;
δ completely characterizes the probability measure Gaussian(δ, 1).
# Arguments:
    law_rpm::tnormal_normal
    n::Int  :  number of probability measures to generate.
"""

function generate_prob_measures(law_rpm::tnormal_normal, n::Int)
    truncated_normal = truncated(Normal(law_rpm.μ, law_rpm.σ), law_rpm.a, law_rpm.b)
    return rand(truncated_normal, n)
end

"""
    generate_prob_measures
Function to generate n values of the δ parameter (not Distribution objects) from discr_normal;
δ completely characterizes the probability measure Gaussian(δ, 1).
As discr_normal is a discrete measure, we sample n values from its atoms with respect to its weights (with replacement).

# Arguments:
    law_rpm::discr_normal
    n::Int  :  number of probability measures to generate.
"""
function generate_prob_measures(law_rpm::discr_normal, n::Int)
    return sample(law_rpm.atoms, Weights(law_rpm.weights), n)
end

"""
    generate_prob_measures

Function to generate n values of the δ parameter (not Distribution objects) from a mixture of two laws of RPM;
δ completely characterizes the probability measure produced by whichever sub-law it was drawn from.
Each δ is generated from either law_rpm_1 or law_rpm_2 with probabilities λ and 1 - λ respectively.

# Arguments:
    law_rpm::mixture
    n::Int  :  number of probability measures to generate.
"""
function generate_prob_measures(law_rpm::mixture, n::Int)
    δs = zeros(n)
    λ = law_rpm.λ # mixing parameter
    for i in 1:n
        if rand() <= λ
            δs[i] = generate_prob_measures(law_rpm.law_rpm_1, 1)[1]
        else
            δs[i] = generate_prob_measures(law_rpm.law_rpm_2, 1)[1]
        end
    end
    return δs
end

"""
    generate_prob_measures

Notifier for laws of RPM that don't have a specific generate_prob_measures method. Raises an informative error.

# Arguments:
    law_rpm::LawRPM
    n::Int  :  number of probability measures to generate.
"""
function generate_prob_measures(law_rpm::LawRPM, n::Int)
    throw(ArgumentError("generate_prob_measures is not defined for $(typeof(law_rpm))."))
end




