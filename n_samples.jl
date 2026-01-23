using Plots
using FLoops # for parallel computing

include("distributions.jl")

include("distances/new_distance.jl")
include("distances/distance_Wasserstein.jl")



function threshold_hipm_nothread(hier_sample_1::HierSample, hier_sample_2::HierSample, n_samples::Int, bootstrap::Bool)
    n = hier_sample_1.n
    m = hier_sample_1.m
    a = minimum((hier_sample_1.a,hier_sample_2.a))
    b = maximum((hier_sample_1.b,hier_sample_2.b))
   
    samples = zeros(n_samples)
    total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
    if bootstrap
        @floop ThreadedEx() for i in 1:n_samples
            indices_1 = sample(1:2*n, n; replace = true)
            indices_2 = sample(1:2*n, n; replace = true)

            new_atoms_1 = total_rows[indices_1,:] # first rows indexed by n random indices to the atoms_1
            new_atoms_2 = total_rows[indices_2,:] # first rows indexed by n random indices to the atoms_2

            new_h_1 = HierSample(new_atoms_1, a, b)
            new_h_2 = HierSample(new_atoms_2, a, b)

            samples[i] = dlip(new_h_1, new_h_2, a, b)
        end
    else
        @floop ThreadedEx() for i in 1:n_samples
            random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

            new_atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
            new_atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
            new_h_1 = HierSample(new_atoms_1,  a, b)
            new_h_2 = HierSample(new_atoms_2,  a, b)

            samples[i] = dlip(new_h_1, new_h_2, a, b)
        end
    end
    return samples
end


K = 2
n_samples = [100, 1000]
n = 100
m = 200
θs = collect(0:0.01:1.0)
bootstrap = false

plots = []

# same pair
q_1 = tnormal_normal(1.0, 1.4, -10.0, 10.0)
q_2 = tnormal_normal(1.0, 1.4, -10.0, 10.0)



for i in 1:K
    h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
    threshold_1 = threshold_hipm_nothread(h_1, h_2, n_samples[1], bootstrap)
    threshold_2 = threshold_hipm_nothread(h_1, h_2, n_samples[2], bootstrap)

    p = plot(θs, quantile(threshold_2, 1 .- θs), label = string(n_samples[2]))
    plot!(p, θs, quantile(threshold_1, 1 .- θs), label = string(n_samples[1]))
    push!(plots, p)
end
