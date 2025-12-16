using BlackBoxOptim
function f(x::Vector{Float64})
    sum(x)
end

res = best_candidate(bboptimize(f; SearchRange = (-1.0, 1.0), NumDimensions = 5))


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