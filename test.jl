
# include("structures.jl")
# using Pkg
# Pkg.activate(".")

# using Distributions, Random
# using Plots

# struct DP<:PPM # Dirichlet process
#     α::Float64
#     p_0::Distribution # function generating observations from p_0
#     a::Float64 
#     b::Float64
#     # R is the observation space
# end


#  function dirichlet_process_without_weight_new(n, α, p_0::Distribution)
#     # auxiliary function
#     # Given function p_0 that returns sample from chosen probability measure P_0
#     # we generate a n-sample from a Dirichlet process with parameter α and p_0

#     @assert n > 0 "n must be positive integer"
#     samples = Vector{Float64}(undef, n)
#     samples[1] = rand(p_0)
#     for i in 2:n
#         if rand() <= α / (α + i - 1)
#             samples[i] = rand(p_0)
#         else
#             index = rand(1:(i-1))
#             samples[i] = samples[index]
#         end
#     end
#     return samples
# end

# # function dirichlet_process_without_weight_new(n, α, p_0::Distribution)
# #     # auxiliary function
# #     # Given function p_0 that returns sample from chosen probability measure P_0
# #     # we generate a n-sample from a Dirichlet process with parameter α and p_0

# #     @assert n > 0 "n must be positive integer"
# #     samples = Vector{Float64}(undef, n)
# #     for i in 1:n
# #         if rand() <= α / (α + i - 1)
# #             samples[i] = rand(p_0)
# #         else
# #             index = rand(1:(i-1))
# #             samples[i] = samples[index]
# #         end
# #     end
# #     return samples
# # end
    


# function dirichlet_process_without_weight(n, α, p_0)
#     # auxiliary function
#     # Given function p_0 that returns sample from chosen probability measure P_0
#     # we generate a n-sample from a Dirichlet process with parameter α and p_0
#     if n == 0
#         return []
#     elseif n == 1 # It's not needed in general but useful to have same seed for iid and exch. case
#         prev = dirichlet_process_without_weight(n-1,α,p_0)
#         return push!(prev, p_0())
#     else 
#         prev = dirichlet_process_without_weight(n-1,α,p_0)
#         if rand() <= α /(α +n-1) # sample from P_0
#             return push!(prev, p_0())
#         else # sample from the already given observations
#             index = rand(1:(n-1))
#             return push!(prev, prev[index])
#         end
#     end  
# end




# α = 10000.0

# n = 1000
# p_0_function = () -> rand(Beta(1,1))
# p_0 = Beta(1,1)


# Random.seed!(1234)

# samples_new = dirichlet_process_without_weight_new(n, α, p_0)
# Random.seed!(1234)
# samples_old = dirichlet_process_without_weight(n, α, p_0_function)

# diff = sum(abs.(samples_new .- samples_old))