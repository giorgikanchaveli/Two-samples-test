include("structures.jl")

using Distributions, Random


struct normal_tnormal<:PPM
    # random probability measure is truncated gaussian distributions on [a,b] with mean generated from normal(μ, σ)
    # sample space is [a,b]

    μ::Float64 # mean of normal distribution from which we generate mean of inner normal distribution
    σ::Float64 # standard deviation of normal distribution from which we generate mean of inner normal distribution
    a::Float64 # interval [a,b] from which atoms are drawn
    b::Float64 # interval [a,b] from which atoms are drawn
end


struct tnormal_normal<:PPM
    # random probability measure is gaussian distributions on R with mean generated from truncated normal(μ, σ) on [a,b]
    # and variance is 1.
    # sample space is R
    μ::Float64 # mean of truncated normal distribution from which we generate mean of inner normal distribution
    σ::Float64 # standard deviation of truncated normal distribution from which we generate mean of inner normal distribution
    a::Float64 # left end of truncation interval
    b::Float64 # right end of truncation interval
end



struct DP<:PPM # Dirichlet process
    α::Float64
    p_0::Distribution # function generating observations from p_0
    a::Float64 
    b::Float64
    # R is the observation space
end

struct normal_normal<:PPM 
    # random probability measure is gaussian distributions with variance equal to 1 and 
    # mean generated from normal(μ, σ) 
    
    # Note that sample space for exhangeable sequences is R

    μ::Float64 # mean of normal distribution from which we generate mean of inner normal distribution
    σ::Float64 # standard deviation of normal distribution from which we generate mean of inner normal distribution
end



struct uniform_normal<:PPM
    # random probability measure is normal distribution with variance 1 and with mean generated from uniform(a,b)
    
    # Note that sample space for exhangeable sequences is R
    a::Float64 
    b::Float64 
end

struct mixture_ppm<:PPM
    # mixture of two laws of random probability measures with mixing parameter λ
    ppm1::PPM
    ppm2::PPM
    λ::Float64 # mixing parameter
end

struct simple_discr_1<:PPM
    # RPM which equals N(-1, 1) with probability 1/2 and N(1,1) with probability 1/2

    # Note that sample space for exhangeable sequences is R
end
struct simple_discr_2<:PPM
    # RPM which equals N(-2, 1) with probability 1/8, N(0,1) with probability 3/4 and N(2,1) with probability 1/8
    
    # Note that sample space for exhangeable sequences is R
end

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


# struct discrrpm<:PPM
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


# struct discr_tnormal<:PPM
#     # discrete measure over n_1 truncated gaussian distributions on [a,b]
#     # sample space is [a,b]

#     n_1::Int # number of normal distributions
#     μ::Vector{Float64} # mean of each normal distribution
#     σ::Vector{Float64} # standard deviation each of normal distribution
#     a::Float64 # interval [a,b] from which atoms are drawn
#     b::Float64 # interval [a,b] from which atoms are drawn
# end





# struct discrete_normal<:PPM
#     # law of RPM is a discrete measure over normal distributions with varying mean but variance equal to 1.

#     μs::Vector{Float64} # Vector of means of normal distributions
#     weights::Vector{Float64} # Vector of weights associated to each normal distribution
# end




# methods to generate hierarhical samples





 function dirichlet_process_without_weight_new(n, α, p_0::Distribution)
    # auxiliary function
    # Given function p_0 that returns sample from chosen probability measure P_0
    # we generate a n-sample from a Dirichlet process with parameter α and p_0

    @assert n > 0 "n must be positive integer"
    samples = Vector{Float64}(undef, n)
    samples[1] = rand(p_0)
    for i in 2:n
        if rand() <= α / (α + i - 1)
            samples[i] = rand(p_0)
        else
            index = rand(1:(i-1))
            samples[i] = samples[index]
        end
    end
    return samples
end

    





function generate_emp(ppm::DP, n_top::Int, n_bottom::Int)
    # given random probability measure from Dirichlet process, generates empirical measure struct
    # which is used for estimating law of rpm.
    # n is the number of random probability measures we want to sample
    # m is the number of atoms from each random probability measure
    atoms = zeros(n_top,n_bottom)
    for i in 1:n_top
        atoms[i,:] = sort(dirichlet_process_without_weight_new(n_bottom, ppm.α, ppm.p_0))
    end
    return emp_ppm(atoms, n_top, n_bottom, ppm.a, ppm.b)
end


function generate_prob_measures(ppm::tnormal_normal, n::Int)
    truncated_normal = truncated(Normal(ppm.μ, ppm.σ), ppm.a, ppm.b)
    return rand(truncated_normal, n)
end

function generate_prob_measures(ppm::simple_discr_1, n::Int)
    means = zeros(n)
    for i in 1:n
        if rand() <= 0.5
            means[i] = -1.0
        else
            means[i] = 1.0
        end
    end 
    return means
end

function generate_prob_measures(ppm::simple_discr_2, n::Int)
    means = zeros(n)
    for i in 1:n
        r = rand()
        if r <= 1/8
            means[i] = -2.0
        elseif r <= 7/8
            means[i] = 0.0
        else
            means[i] = 2.0
        end
    end 
    return means
end


function generate_prob_measures(ppm::mixture_ppm, n::Int)
    means = zeros(n)
    λ = ppm.λ
    for i in 1:n
        if rand() <= λ
            means[i] = generate_prob_measures(ppm.ppm1, 1)[1]
        else
            means[i] = generate_prob_measures(ppm.ppm2, 1)[1]
        end
    end
    return means
end




function generate_emp(ppm::tnormal_normal, n_top::Int, n_bottom::Int)

    atoms = zeros(n_top,n_bottom)
    for i in 1:n_top
        truncated_normal = truncated(Normal(ppm.μ, ppm.σ), ppm.a, ppm.b)
        μ_inner = rand(truncated_normal)
        
        atoms[i,:] = sort(rand(Normal(μ_inner, 1.0), n_bottom))
    end
    a = @views minimum(atoms[:,1])
    b = @views maximum(atoms[:,end])
    return emp_ppm(atoms, n_top, n_bottom, a, b)
end


function generate_emp(ppm::normal_tnormal, n_top::Int, n_bottom::Int)
    # given random probability measure which takes values as trunacted normal distributions where mean is random variable from normal(μ,σ), 
    # we generate hieraarchical measure
    # which is used for estimating law of rpm.

    # n is the number of random probability measures we want to sample
    # m is the number of atoms from each random probability measure
    a,b = ppm.a, ppm.b
    atoms = zeros(n_top,n_bottom)
    for i in 1:n_top
        μ_inner = rand(Normal(ppm.μ, ppm.σ))
        truncated_normal = truncated(Normal(μ_inner, ppm.σ), a, b)
        atoms[i,:] = sort(rand(truncated_normal, n_bottom))
    end
    return emp_ppm(atoms, n_top, n_bottom, a, b)
end


function issorted(m::Matrix{Float64})
    flag = true
    for i in m.size[1]
        row = m[i,:]
        for i in 1:(length(row)-1)
            if row[i] > row[i + 1]
                flag = false
                return flag
            end
        end
    end
    return flag
end



function generate_emp(ppm::mixture_ppm, n_top::Int, n_bottom::Int)
    atoms = zeros(n_top,n_bottom)
    λ = ppm.λ
    for i in 1:n_top
        if rand() <= λ
            atoms[i,:] = generate_emp(ppm.ppm1, 1, n_bottom).atoms[1,:]
        else
            atoms[i,:] = generate_emp(ppm.ppm2, 1, n_bottom).atoms[1,:]
        end
    end
    a = @views minimum(atoms[:,1])
    b = @views maximum(atoms[:,end])
    @assert issorted(atoms)
    return emp_ppm(atoms, n_top, n_bottom, a, b)
end




function generate_emp(ppm::simple_discr_1, n_top::Int, n_bottom::Int)
    atoms = zeros(n_top,n_bottom)
    for i in 1:n_top
        if rand() <= 0.5
            atoms[i,:] = sort(rand(Normal(-1.0,1.0), n_bottom))
        else
            atoms[i,:] = sort(rand(Normal(1.0,1.0), n_bottom))
        end
    end
    a = @views minimum(atoms[:,1])
    b = @views maximum(atoms[:,end])
    return emp_ppm(atoms, n_top, n_bottom, a, b)
end

function generate_emp(ppm::simple_discr_2, n_top::Int, n_bottom::Int)
    atoms = zeros(n_top,n_bottom)
    for i in 1:n_top
        r = rand()
        if r <= 1/8
            atoms[i,:] = sort(rand(Normal(-2.0,1.0), n_bottom))
        elseif r <= 7/8
            atoms[i,:] = sort(rand(Normal(0.0,1.0), n_bottom))
        else
            atoms[i,:] = sort(rand(Normal(2.0,1.0), n_bottom))
        end
    end
    a = @views minimum(atoms[:,1])
    b = @views maximum(atoms[:,end])
    return emp_ppm(atoms, n_top, n_bottom, a, b)
end



# function generate_emp(ppm::discrrpm, n_top::Int, n_bottom::Int)
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
#     return emp_ppm(atoms, n_top, n_bottom, a, b)
# end


# function generate_emp(ppm::discr_tnormal, n_top::Int, n_bottom::Int)
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
#     return emp_ppm(atoms, n_top, n_bottom, a, b)
# end



# function generate_emp(pms::Vector{Normal}, n_top::Int, n_bottom::Int)
#     # given vector of n_top normal distributions, we generate n_bottom observations from each of them.
#     # length sequence of observations. 
#     @assert length(pms) == n_top "n and length of vector of probability measures are not equal"
#     atoms = zeros(n_top,n_bottom)
#     for i in 1:n_top
#         atoms[i,:] = rand(pms[i], n_bottom)
#     end
#     a = minimum(atoms) # left end of an interaval where observations take values
#     b = maximum(atoms) # right end of an interaval where observations take values
#     return emp_ppm(atoms, n_top, n_bottom, a, b)
# end

# function generate_emp(pms::Vector{Any}, n_top::Int, n_bottom::Int)
#     # given vector of n_top any probability measures, we generate n_bottom observations from each of them.

#     # Each element in pms vector must be of type distributino ? 

#     @assert length(pms) == n_top "n and length of vector of probability measures are not equal"
#     atoms = zeros(n_top,n_bottom)
#     for i in 1:n_top
#         atoms[i,:] = rand(pms[i], n_bottom)
#     end
#     a = minimum(atoms) # left end of an interaval where observations take values
#     b = maximum(atoms) # right end of an interaval where observations take values
#     return emp_ppm(atoms, n_top, n_bottom, a, b)
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







# function generate_emp(ppm::PPM, n_top::Int, n_bottom::Int)
#     # given law of random probability measure, instead of directly generating hierarchical samples, we firstly
#     # generate probability meassures and then observations from them.
#     pms = generate_prob_measures(ppm, n_top)
#     return generate_emp(pms, n_top, n_bottom)
# end








