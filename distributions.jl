include("structures.jl")

### Dirichlet Process


using Random


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

function cdf_same(x)
    if x < -1/2
        return 0
    elseif x < 1/2
        return (x + 1/2)
    else
        return 1
    end
end

function cdf_splitting(x)
    if x < -1
        return 0
    elseif x < -3/4
        return 2*(x+1)
    elseif x < 3/4
        return 1/2
    elseif x < 1
        return 1/2 + + 2*(x - 3/4)
    else
        return 1
    end
end


# IID

struct Discrpm <: PM
    # discrete probability measure
    # sample space is [a,b]
    atoms::Vector{Float64}
    weights::Vector{Float64}
    a::Float64
    b::Float64

    function Discrpm(n_atoms::Int, a::Float64, b::Float64)
        # given n_atoms constructs discrete measure which has atoms in [a,b] and weights
        atoms = rand(Uniform(a,b), n_atoms)
        weights = rand(n_atoms)
        weights = weights ./ sum(weights)
        new(atoms,weights,a,b)
    end

    function Discrpm(atoms::Vector{Float64}, weights::Vector{Float64},a::Float64,b::Float64)
        new(atoms, weights,a,b)
    end

end

struct Uniformpm <: PM
    # uniform probability measure
    # sample space is [a,b]
    a::Float64
    b::Float64

    function Uniformpm(a::Float64,b::Float64)
        new(a,b)
    end
end

struct samepm <: PM
    # uniform probability measure on [-1,1] from above function "probability"
    # sample space is [a,b]
    a::Float64
    b::Float64

    function samepm(a::Float64,b::Float64)
        new(a,b)
    end
end

struct splittingpm <: PM # for above function "probability"
    # uniform probability measure
    # sample space is [a,b]
    a::Float64
    b::Float64

    function splittingpm(a::Float64,b::Float64)
        new(a,b)
    end
end 

struct Betapm <: PM
    # beta probability measure
    # sample space is [a,b]
    a::Float64
    b::Float64
    α::Float64  
    β::Float64
    function Betapm(a::Float64,b::Float64,α::Float64,β::Float64)
        new(a,b,α,β)
    end
end



# from each probability measure we want method to generate empirical probability measure


function generate_emp(p::Discrpm, n::Int)
    # generates empirical probability measure from a given discrete probability measure
    # p: probability measure
    # n: number of samples
    atoms = p.atoms
    r = Categorical(p.weights)
    return emp_pm(atoms[rand(r, n)], p.a, p.b)
end

function generate_emp(p::Uniformpm, n::Int)
    # generates empirical probability measure from a given uniform probability measure
    # p: probability measure
    # n: number of samples
    return emp_pm(rand(Uniform(p.a,p.b),n), p.a, p.b)
end

function generate_emp(p::Betapm, n::Int)
    # generates empirical probability measure from a given beta probability measure
    # p: probability measure
    # n: number of samples
    return emp_pm(rand(Beta(p.α,p.β),n), p.a, p.b)
end 

function generate_emp(p::samepm, n::Int)
    # generates empirical probability measure from a given uniform(-1,1) probability measure
    # p: probability measure
    # n: number of samples
    return emp_pm([probability("same") for i in 1:n], p.a, p.b)
end

function generate_emp(p::splittingpm, n::Int)
    # generates empirical probability measure from a given splitting (see above) probability measure
    # p: probability measure
    # n: number of samples
    return emp_pm([probability("splitting") for i in 1:n], p.a, p.b)
end








# Exchangeable


struct DP<:PPM # Dirichlet process
    α::Float64
    p_0::Function # function generating observations from p_0
    a::Float64 
    b::Float64
    # [a,b] is the observation space
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





