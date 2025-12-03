
using Distributions, Random
using BenchmarkTools

include("distributions.jl")
include("structures.jl")
include("distances/distance_Wasserstein.jl")
include("distances/new_distance.jl")







# parameters for functions

n = 100
m = 200

# a = 0.0
# b = 1.0


q_1 = tnormal_normal(1.0,1.0,-10.0,10.0)
q_2 = tnormal_normal(1.0,1.0,-10.0,10.0)
q_3 = mixture_ppm(q_1, q_2, 0.5)
q_4 = DP(1.0, Beta(1,1), 0.0,1.0)


h_1,h_2 = generate_emp(q_1,n,m),  generate_emp(q_3,n,m)
a = minimum((h_1.a, h_2.a))
b = maximum((h_1.b, h_2.b))
# @time [old_value = dlip(h_1,h_2, a, b) for i in 1:2]
# @time [dlip_new(h_1.atoms,h_2.atoms,a,b) for i in 1:2]

println("Benchmarking dlip (Original Measure Format - 3D Array):")
# Measures the fully compiled time accurately
@btime dlip(h_1, h_2, a, b)

# println("\nBenchmarking dlip_new (Optimized Vector Measure Format - 2D Array):")
# # Measures the fully compiled time accurately
# @btime dlip_new(h_1.atoms, h_2.atoms, a, b)