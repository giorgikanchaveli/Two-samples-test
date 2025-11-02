println(Threads.nthreads())
using Base.Threads: @threads


include("distances/new_distance.jl")
include("distances/distance_Wasserstein.jl")
include("distributions.jl")


decision_dlip = function(q_1::PPM, q_2::PPM, n::Int, m::Int, θ::Float64, n_sim::Int, n_samples::Int, boostrap = true)
    hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)

    distance_observed = dlip(hier_sample_1, hier_sample_2)

    # obtain threshold

    if boostrap
        total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
        boostrap_samples = zeros(n_samples)
        for k in 1:n_samples
            indices_1 = sample(1:2*n, n; replace = true)
            indices_2 = sample(1:2*n, n; replace = true)
            atoms_1 = total_rows[indices_1,:]  # resample from pooled hierarchical sample
            atoms_2 = total_rows[indices_2,:]  # resample from pooled hierarchical sample
            
        
            hier_sample_1_boostrap = emp_ppm(atoms_1, n, m, 0.0, 1.0)
            hier_sample_2_boostrap = emp_ppm(atoms_2, n, m, 0.0, 1.0)

            boostrap_samples[k] = dlip(hier_sample_1_boostrap, hier_sample_2_boostrap)
        end

    threshold = quantile(boostrap_samples, 1 - θ)
    return 1 * (distance_observed > threshold)
    end
    permuted_samples = zeros(n_samples) # store samples of distances
    total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
        for k in 1:n_samples
            random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures
            atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n_top random indices to the atoms_1
            atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n_top random indices to the atoms_2

            hier_sample_1_permuted = emp_ppm(atoms_1, n, m, 0.0, 1.0)
            hier_sample_2_permuted = emp_ppm(atoms_2, n, m, 0.0, 1.0)

            
            permuted_samples[k] = dlip(hier_sample_1_permuted, hier_sample_2_permuted)
        end
    threshold = quantile(permuted_samples, 1 - θ)
    return 1 * (distance_observed > threshold)
    
end


p_0 = () -> rand(Beta(1,1))
p_1 = () -> rand(Beta(1,1.1))

alpha_1 = 10.0
alpha_2 = 11.5

q_1 = DP(alpha_1, p_0, 0.0, 1.0)
q_2 = DP(alpha_1, p_1, 0.0, 1.0)



n = 40
m = 40

θ = 0.05
n_sim = 10
n_samples = 1

decision_dlip(q_1,q_2,n, m, θ, n_sim, n_samples, true)


function f(i)
    for j in 1:i
        for k in 1:i
            for l in 1:i
                z = k + l + j
                for s in 1:i
                    q = z + 3
                end
            end
        end
    end
end

# Sequential
t_serial = @elapsed begin
    [f(i) for i in 1:10000]
end

# Parallel
t_parallel = @elapsed begin
    Threads.@threads for i in 1:10000
        f(i)
    end
end

println("Serial time:   ", t_serial)
println("Parallel time: ", t_parallel)