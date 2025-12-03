
using Distributions, Random
using BenchmarkTools

using FLoops

include("distributions.jl")
include("structures.jl")
include("distances/distance_Wasserstein.jl")
include("distances/new_distance.jl")
using FLoops # for parallel computing



function threshold_wow_old(hier_sample_1::emp_ppm, hier_sample_2::emp_ppm, θ::Float64, n_samples::Int, bootstrap::Bool)
    n = hier_sample_1.n

    atoms_1 = sort(hier_sample_1.atoms, dims = 2)
    atoms_2 = sort(hier_sample_2.atoms, dims = 2)

    samples = zeros(n_samples)
    total_rows = vcat(atoms_1, atoms_2) # collect all rows
    if bootstrap
        for i in 1:n_samples
            indices_1 = sample(1:2*n, n; replace = true)
            indices_2 = sample(1:2*n, n; replace = true)

            new_atoms_1 = total_rows[indices_1,:] # first rows indexed by n random indices to the atoms_1
            new_atoms_2 = total_rows[indices_2,:] # first rows indexed by n random indices to the atoms_2

            samples[i] = ww(new_atoms_1, new_atoms_2) # sorted = true
        end
    else
        for i in 1:n_samples
            random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

            new_atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
            new_atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
         
            samples[i] = ww(new_atoms_1, new_atoms_2) # sorted = true
        end
    end
    return quantile(samples, 1 - θ)
end



function threshold_wow_new(hier_sample_1::emp_ppm, hier_sample_2::emp_ppm, θ::Float64, n_samples::Int, bootstrap::Bool)
    n = hier_sample_1.n

    atoms_1 = sort(hier_sample_1.atoms, dims = 2)
    atoms_2 = sort(hier_sample_2.atoms, dims = 2)

    samples = zeros(n_samples)
    total_rows = vcat(atoms_1, atoms_2) # collect all rows
    if bootstrap
        @floop ThreadedEx() for i in 1:n_samples
            indices_1 = sample(1:2*n, n; replace = true)
            indices_2 = sample(1:2*n, n; replace = true)

            new_atoms_1 = total_rows[indices_1,:] # first rows indexed by n random indices to the atoms_1
            new_atoms_2 = total_rows[indices_2,:] # first rows indexed by n random indices to the atoms_2

            samples[i] = ww(new_atoms_1, new_atoms_2) # sorted = true
        end
    else
        @floop ThreadedEx() for i in 1:n_samples
            random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

            new_atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
            new_atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
         
            samples[i] = ww(new_atoms_1, new_atoms_2) # sorted = true
        end
    end
    return quantile(samples, 1 - θ)
end








# parameters for functions

n = 100
m = 200

# a = 0.0
# b = 1.0

n_samples = 100

q_1 = tnormal_normal(1.0,1.0,-10.0,10.0)
q_2 = tnormal_normal(1.0,1.0,-10.0,10.0)
q_3 = mixture_ppm(q_1, q_2, 0.5)
q_4 = DP(1.0, Beta(1,1), 0.0,1.0)

h_1,h_2 = generate_emp(q_4,n,m),  generate_emp(q_4,n,m)


Random.seed!(1234)
println("\nBenchmarking new")

@btime threshold_wow_new(h_1,h_2,0.05,n_samples,true)

Random.seed!(1234)
println("Benchmarking old")

@btime threshold_wow_old(h_1,h_2,0.05,n_samples,true)


# @time [old_value = dlip(h_1,h_2, a, b) for i in 1:2]
# @time [dlip_new(h_1.atoms,h_2.atoms,a,b) for i in 1:2]

#println("Benchmarking dlip (Original Measure Format - 3D Array):")
# Measures the fully compiled time accurately
#@btime dlip(h_1, h_2, a, b)

# println("\nBenchmarking dlip_new (Optimized Vector Measure Format - 2D Array):")
# # Measures the fully compiled time accurately
# @btime dlip_new(h_1.atoms, h_2.atoms, a, b)