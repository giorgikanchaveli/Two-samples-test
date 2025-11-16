using RCall # to call R functions
using Plots

include("distributions.jl")

include("distances/new_distance.jl")
include("distances/distance_Wasserstein.jl")
using DataFrames
using CSV




function rejection_rate_hipm_boostrap(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)
    rej_rate = 0.0

    for s in 1:S
        hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        a = minimum([hier_sample_1.a, hier_sample_2.a])
        b = maximum([hier_sample_1.b, hier_sample_2.b])
        hier_sample_1.a = a
        hier_sample_2.a = a
        hier_sample_1.b = b
        hier_sample_2.b = b
        observed_test_stat = dlip(hier_sample_1, hier_sample_2)
        
        # obtain quantile using bootstrap approach
        boostrap_samples = zeros(n_boostrap) # zeros can be improved
        
        total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
        for i in 1:n_boostrap
            indices_1 = sample(1:2*n, n; replace = true)
            indices_2 = sample(1:2*n, n; replace = true)
            atoms_1 = total_rows[indices_1,:]  # resample from pooled hierarchical sample
            atoms_2 = total_rows[indices_2,:]  # resample from pooled hierarchical sample
            
        
            hier_sample_1_boostrap = emp_ppm(atoms_1, n, m, a, b)
            hier_sample_2_boostrap = emp_ppm(atoms_2, n, m, a, b)

            boostrap_samples[i] = dlip(hier_sample_1_boostrap, hier_sample_2_boostrap)
        end
        threshold = quantile(boostrap_samples, 1 - θ)
        
        rej_rate += 1.0*(observed_test_stat > threshold)
    end
    return rej_rate / S
end

function rejection_rate_hipm_permutation(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)
    rej_rate = 0.0

    for s in 1:S
        hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        a = minimum([hier_sample_1.a, hier_sample_2.a])
        b = maximum([hier_sample_1.b, hier_sample_2.b])
        hier_sample_1.a = a
        hier_sample_2.a = a
        hier_sample_1.b = b
        hier_sample_2.b = b
        observed_test_stat = dlip(hier_sample_1, hier_sample_2)
        
        # obtain quantile using permutation approach
        permutation_samples = zeros(n_permutation) # zeros can be improved
        
        total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
        for i in 1:n_permutation
            random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

            atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
            atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
            hier_sample_1_permutation = emp_ppm(atoms_1, n, m, a, b)
            hier_sample_2_permutation = emp_ppm(atoms_2, n, m, a, b)

            permutation_samples[i] = dlip(hier_sample_1_permutation, hier_sample_2_permutation)
        end
        threshold = quantile(permutation_samples, 1 - θ)
        
        rej_rate += 1.0*(observed_test_stat > threshold)
    end
    return rej_rate / S
end


function rejection_rate_wow_boostrap(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)
    rej_rate = 0.0

    for s in 1:S
        hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        a = minimum([hier_sample_1.a, hier_sample_2.a])
        b = maximum([hier_sample_1.b, hier_sample_2.b])
        hier_sample_1.a = a
        hier_sample_2.a = a
        hier_sample_1.b = b
        hier_sample_2.b = b
        observed_test_stat = ww(hier_sample_1, hier_sample_2)
        
        # obtain quantile using bootstrap approach
        boostrap_samples = zeros(n_boostrap) # zeros can be improved
  
        total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
        for i in 1:n_boostrap
            indices_1 = sample(1:2*n, n; replace = true)
            indices_2 = sample(1:2*n, n; replace = true)
            atoms_1 = total_rows[indices_1,:]  # resample from pooled hierarchical sample
            atoms_2 = total_rows[indices_2,:]  # resample from pooled hierarchical sample
            
        
            hier_sample_1_boostrap = emp_ppm(atoms_1, n, m, a, b)
            hier_sample_2_boostrap = emp_ppm(atoms_2, n, m, a, b)

            boostrap_samples[i] = ww(hier_sample_1_boostrap, hier_sample_2_boostrap)
        end
        threshold = quantile(boostrap_samples, 1 - θ)
        
        rej_rate += 1.0*(observed_test_stat > threshold)
    end
    return rej_rate / S
end

function rejection_rate_wow_permutation(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)
    rej_rate = 0.0

    for s in 1:S
        hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        a = minimum([hier_sample_1.a, hier_sample_2.a])
        b = maximum([hier_sample_1.b, hier_sample_2.b])
        hier_sample_1.a = a
        hier_sample_2.a = a
        hier_sample_1.b = b
        hier_sample_2.b = b
        observed_test_stat = ww(hier_sample_1, hier_sample_2)
        
        # obtain quantile using permutation approach
        permutation_samples = zeros(n_permutation) # zeros can be improved
        
        total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
        for i in 1:n_permutation
            random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

            atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
            atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
            hier_sample_1_permutation = emp_ppm(atoms_1, n, m, a, b)
            hier_sample_2_permutation = emp_ppm(atoms_2, n, m, a, b)

            permutation_samples[i] = ww(hier_sample_1_permutation, hier_sample_2_permutation)
        end
        threshold = quantile(permutation_samples, 1 - θ)
        
        rej_rate += 1.0*(observed_test_stat > threshold)
    end
    return rej_rate / S
end

