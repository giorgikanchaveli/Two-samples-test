# I see two ways to extend hipm for different sample sizes in two groups. 

# 1) numerically compute gradient and use the original code ( I guess substituting evaluateobjective function)

# 2) just define objective function in julia and then optimize on box.

include("structures.jl")
include("distances/hipm.jl")
include("distances/distance_Wasserstein.jl")
using LinearAlgebra
using BlackBoxOptim
using ExactOptimalTransport
using Tulip





function dlip_diffsize(atoms_1::Matrix{Float64}, atoms_2::Matrix{Float64}, 
              weights_1::Matrix{Float64}, weights_2::Matrix{Float64}, a::Float64, b::Float64,
              nGrid::Int = 250, maxTime = 10.0::Float64)
    s1 = size(atoms_1)
    s2 = size(atoms_2)

    deltaX = (b-a) / nGrid


    # Project atoms on a grid
    weight_atoms_1 = zeros(s1[1], nGrid + 1)
    weight_atoms_2 = zeros(s2[1], nGrid + 1)

    for i in 1:s1[1]
        weight_atoms_1[i, :] .= projectionGridMeasure_new(atoms_1[i, :], weights_1[i,:], a,b,nGrid)
    end
    for i in 1:s2[1]
        weight_atoms_2[i, :] .= projectionGridMeasure_new(atoms_2[i, :], weights_2[i,:], a,b,nGrid)
    end

    # define objective function to optimize

    function obj(g::Vector{Float64})
        # we define firstly f from g and then wass distance 
        @assert length(g) == nGrid
        f = deltaX .* vcat([0.0], cumsum(g))
        
        weights_mu = fill(1.0 / s1[1], s1[1])
        weights_nu = fill(1.0 / s2[1], s2[1])
        
        atoms_mu = weight_atoms_1 * f
        atoms_nu = weight_atoms_2 * f

        C = abs.(atoms_mu .- transpose(atoms_nu))

        gamma = ExactOptimalTransport.emd(weights_mu, weights_nu, C, Tulip.Optimizer() )
       
        # @timeit timer "compute transport cost" 
        #output = sum( gamma .* C )
        output = -1.0 * sum(gamma.*C)
        return output
    end

    res = bboptimize(obj; SearchRange = (-1.0, 1.0), NumDimensions = nGrid, MaxTime = maxTime)
    return -1.0 * best_fitness(res)
end




n = 70
m = 100
atoms_1 = rand(n,m)
atoms_2 = rand(n,m)

weights_1 = fill(1.0 / m, n, m)
weights_2 = fill(1.0 / m, n, m)


# weights_1 = rand(n, m)
# weights_2 = rand(n, m)

# for i in 1:n
#     s1 = sum(weights_1[i, :])
#     s2 = sum(weights_2[i, :])
#     weights_1[i,:] ./= s1 
#     weights_2[i,:] ./= s2 
# end
a = 0.0
b = 1.0
maxTime = 500.0

hipm_old = dlip(atoms_1, atoms_2,  a, b)
hipm_new = dlip_diffsize(atoms_1, atoms_2, weights_1, weights_2, a, b, 250, maxTime)
wwass = ww(atoms_1, atoms_2)
println("ww : $(wwass)")
println("HIPM Old : $(hipm_old)")
println("HIPM New : $(hipm_new)")







# using BlackBoxOptim
# function f(x::Vector{Float64})
#     sum(x)
# end













#res = bboptimize(f; SearchRange = (-1.0, 1.0), NumDimensions = 5)


# # 1. Define the Multivariate Function (same as before)
# function rosenbrock_3d(x)
#     return 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2 + 100 * (x[3] - x[2]^2)^2 + (x[2] - 1)^2
# end

# # 2. Define the Search Range
# # The bounds [-1, 1]^n are defined as a vector of tuples.
# n = 3
# search_range = [(-1.0, 1.0) for _ in 1:n] # [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]

# # 3. Run the Global Optimization
# results = bboptimize(
#     rosenbrock_3d;
#     SearchRange = search_range,
#     Method = :adaptive_de_rand_1_bin_advanced, # A robust default method
#     MaxTime = 60.0 # Stop after 60 seconds (or use MaxSteps)
# )

# # 4. Extract the Results
# optimal_x = best_candidate(results)
# minimum_value = best_fitness(results)

# println("Optimal x (Global): ", optimal_x)
# println("Minimum value (Global): ", minimum_value)
