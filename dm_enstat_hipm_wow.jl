# Here we compare 4 testing schemes: Dubey & Muller, Energy statistic (Szekely & Rizzo 2004), HIPM, WoW.

# In particular we compare the rejection rates on the example of random probability measures from Fig 1 Dubey & Muller.


# ### Plots

# 1. **Varying Mean — Permutation (Figure 1 Left)
   
#    We consider Truncated normal - normal model for the law of random probability measures. 
#                                   In that model, firstly mean $\widetilde{\mu}$ is generated from TN ditribution and 
#                                   random probability measure is normal distribution with mean $\widetilde{\mu}$ and 
#                                   variance $1.$ We consider several pairs of such laws by varying the means of prior distribution.
#                                   For each pair, we display the rejection rates. For HIPM and WoW s we use Permutation approach.

# 2. **Varying Mean — Bootstrap (Figure 1 Left)
 
#    Same setup as (1), but using the **bootstrap** approach for **HIPM** and **WoW**.

# 3. **Varying Variance — Permutation (Figure 1 Right)**  
   
#    Here difference from 1) and 2) is that we vary variance of the prior distribution of random probability measure.

# 4. **Varying Variance — Bootstrap (Figure 1 Right)**  

#    Same as 3) but with Boostrap approach for HIPM and WoW.

using RCall # to call R functions
using Plots

include("distributions.jl")

include("distances/new_distance.jl")
include("distances/distance_Wasserstein.jl")
using DataFrames
using CSV
using FLoops



function test_statistic_energy(hier_sample_1::emp_ppm, hier_sample_2::emp_ppm)
    n = hier_sample_1.n
    atoms_1, atoms_2 = hier_sample_1.atoms, hier_sample_2.atoms
    distances_x = Matrix{Float64}(undef, n, n)
    distances_xy = Matrix{Float64}(undef, n, n)
    distances_y = Matrix{Float64}(undef, n, n)

    for i in 1:n
        for j in 1:n
            distances_x[i, j] = wasserstein1DUniform(atoms_1[i,:], atoms_1[j,:], 1)
            distances_xy[i, j] = wasserstein1DUniform(atoms_1[i,:], atoms_2[j,:], 1)
            distances_y[i, j] = wasserstein1DUniform(atoms_2[i,:], atoms_2[j,:], 1)
        end
    end
    distance = 2 * mean(distances_xy) - mean(distances_x) - mean(distances_y)
    return distance * n / 2
end



function rejection_rate_energy_boostrap(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)
    rej_rate = 0.0

    for s in 1:S
        hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        observed_test_stat = test_statistic_energy(hier_sample_1, hier_sample_2)
        
        # obtain quantile using bootstrap approach
        boostrap_samples = zeros(n_boostrap) # zeros can be improved
        a = minimum([hier_sample_1.a, hier_sample_2.a])
        b = maximum([hier_sample_1.b, hier_sample_2.b])
        total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
        for i in 1:n_boostrap
            indices_1 = sample(1:2*n, n; replace = true)
            indices_2 = sample(1:2*n, n; replace = true)
            atoms_1 = total_rows[indices_1,:]  # resample from pooled hierarchical sample
            atoms_2 = total_rows[indices_2,:]  # resample from pooled hierarchical sample
            
        
            hier_sample_1_boostrap = emp_ppm(atoms_1, n, m, a, b)
            hier_sample_2_boostrap = emp_ppm(atoms_2, n, m, a, b)

            boostrap_samples[i] = test_statistic_energy(hier_sample_1_boostrap, hier_sample_2_boostrap)
        end
        threshold = quantile(boostrap_samples, 1 - θ)
        
        rej_rate += 1.0*(observed_test_stat > threshold)
    end
    return rej_rate / S
end


function rejection_rate_energy_boostrap_parallel(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)
    rej_rate = 0.0

    @floop ThreadedEx() for s in 1:S
        hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        observed_test_stat = test_statistic_energy(hier_sample_1, hier_sample_2)
        
        # obtain quantile using bootstrap approach
        boostrap_samples = zeros(n_boostrap) # zeros can be improved
        a = minimum([hier_sample_1.a, hier_sample_2.a])
        b = maximum([hier_sample_1.b, hier_sample_2.b])
        total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
        for i in 1:n_boostrap
            indices_1 = sample(1:2*n, n; replace = true)
            indices_2 = sample(1:2*n, n; replace = true)
            atoms_1 = total_rows[indices_1,:]  # resample from pooled hierarchical sample
            atoms_2 = total_rows[indices_2,:]  # resample from pooled hierarchical sample
            
        
            hier_sample_1_boostrap = emp_ppm(atoms_1, n, m, a, b)
            hier_sample_2_boostrap = emp_ppm(atoms_2, n, m, a, b)

            boostrap_samples[i] = test_statistic_energy(hier_sample_1_boostrap, hier_sample_2_boostrap)
        end
        threshold = quantile(boostrap_samples, 1 - θ)
        
        @reduce rej_rate += 1.0*(observed_test_stat > threshold)
    end
    return rej_rate / S
end





function rejection_rate_dm_boostrap(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)

    rej_rate = 0.0
    
 
    for s in 1:S
     
        hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        atoms_1, atoms_2 = hier_sample_1.atoms, hier_sample_2.atoms

        @rput atoms_1 atoms_2 n n_boostrap
        R"""
        # if (!requireNamespace("frechet", quietly = TRUE)) {
        #   install.packages("frechet", repos="https://cloud.r-project.org")
        # }
        library(frechet)
        atoms_all = rbind(atoms_1, atoms_2)

        group <- c(rep(1, n), rep(2, n))

        result_denanova = DenANOVA(
            yin = atoms_all,
            group = group,
            optns = list(boot = TRUE, R = n_boostrap)
        )
        pvalue = result_denanova$pvalBoot
        """
        @rget pvalue
        rej_rate += 1 * (pvalue < θ)
    end
    rej_rate /= S
    return rej_rate
end


function rejection_rate_dm_boostrap_parallel(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)

    rej_rate = 0.0
    
 
    @floop ThreadedEx() for s in 1:S
     
        hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        atoms_1, atoms_2 = hier_sample_1.atoms, hier_sample_2.atoms

        @rput atoms_1 atoms_2 n n_boostrap
        R"""
        # if (!requireNamespace("frechet", quietly = TRUE)) {
        #   install.packages("frechet", repos="https://cloud.r-project.org")
        # }
        library(frechet)
        atoms_all = rbind(atoms_1, atoms_2)

        group <- c(rep(1, n), rep(2, n))

        result_denanova = DenANOVA(
            yin = atoms_all,
            group = group,
            optns = list(boot = TRUE, R = n_boostrap)
        )
        pvalue = result_denanova$pvalBoot
        """
        @rget pvalue
        @reduce rej_rate += 1 * (pvalue < θ)
    end
    rej_rate /= S
    return rej_rate
end



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

function rejection_rate_hipm_boostrap_parallel(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)
    rej_rate = 0.0

    @floop ThreadedEx() for s in 1:S
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
        
        @reduce rej_rate += 1.0*(observed_test_stat > threshold)
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

function rejection_rate_hipm_permutation_parallel(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)
    rej_rate = 0.0

    @floop ThreadedEx() for s in 1:S
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
        
        @reduce rej_rate += 1.0*(observed_test_stat > threshold)
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

function rejection_rate_wow_boostrap_parallel(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)
    rej_rate = 0.0

    @floop ThreadedEx() for s in 1:S
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
        
        @reduce rej_rate += 1.0*(observed_test_stat > threshold)
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



function rejection_rate_wow_permutation_parallel(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)
    rej_rate = 0.0

    @floop ThreadedEx() for s in 1:S
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
        
        @reduce rej_rate += 1.0*(observed_test_stat > threshold)
    end
    return rej_rate / S
end





function save_varying_mean_boostrap(n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)
    δs = collect(-1.0:2.0:1.0)

    rej_rates_hipm = zeros(length(δs))
    rej_rates_wow = zeros(length(δs))
    rej_rates_dm = zeros(length(δs))
    rej_rates_energy = zeros(length(δs))

    for (i, δ) in enumerate(δs)
        μ_1, σ_1, a, b = 0.0, 0.5, -10.0, 10.0
        μ_2, σ_2, a, b = δ, 0.5, -10.0, 10.0

        q_1 = tnormal_normal(μ_1, σ_1, a, b)
        q_2 = tnormal_normal(μ_2, σ_2, a, b)

        rej_rates_hipm[i] = rejection_rate_hipm_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
        rej_rates_dm[i] = rejection_rate_dm_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
        rej_rates_wow[i] = rejection_rate_wow_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
        rej_rates_energy[i] = rejection_rate_energy_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
    end

    varying_mean_boostrap = plot(title = "Rejection rates of 4 testing schemes, boostrap", xlabel = "δ", ylabel = "Rej rate", xlims=(-1.0, 1.1), ylims = (-0.1, 1.1))
    plot!(varying_mean_boostrap, δs, rej_rates_dm, label = "dm", color = "red",marker = (:circle, 4))
    plot!(varying_mean_boostrap, δs, rej_rates_hipm, label = "hipm", color = "green", marker = (:circle, 4))
    plot!(varying_mean_boostrap, δs, rej_rates_wow, label = "wow", color = "brown", marker = (:circle, 4))
    plot!(varying_mean_boostrap, δs, rej_rates_energy, label = "Energy", color = "blue", marker = (:circle, 4))
    filepath = joinpath(pwd(), "frechet/figure1")
    savefig(varying_mean_boostrap,joinpath(filepath, "varying_mean_n=$(n)_m=$(m)_S=$(S)_nboostrap=$(n_boostrap).png"))
end




function save_varying_mean_permutation(n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)
    # note that we still use boostrap for energy and dm
    δs = collect(-1.0:2.0:1.0)

    rej_rates_hipm = zeros(length(δs))
    rej_rates_wow = zeros(length(δs))
    rej_rates_dm = zeros(length(δs))
    rej_rates_energy = zeros(length(δs))

    for (i, δ) in enumerate(δs)
        μ_1, σ_1, a, b = 0.0, 0.5, -10.0, 10.0
        μ_2, σ_2, a, b = δ, 0.5, -10.0, 10.0

        q_1 = tnormal_normal(μ_1, σ_1, a, b)
        q_2 = tnormal_normal(μ_2, σ_2, a, b)

        rej_rates_hipm[i] = rejection_rate_hipm_permutation(q_1, q_2, n, m, S, θ, n_permutation)
        rej_rates_dm[i] = rejection_rate_dm_boostrap(q_1, q_2, n, m, S, θ, n_permutation)
        rej_rates_wow[i] = rejection_rate_wow_permutation(q_1, q_2, n, m, S, θ, n_permutation)
        rej_rates_energy[i] = rejection_rate_energy_boostrap(q_1, q_2, n, m, S, θ, n_permutation)
    end

    varying_mean_permutation = plot(title = "Rejection rates of 4 testing schemes, permutation", xlabel = "δ", ylabel = "Rej rate", xlims=(-1.0, 1.1), ylims = (-0.1, 1.1))
    plot!(varying_mean_permutation, δs, rej_rates_dm, label = "dm", color = "red", marker = (:circle, 4))
    plot!(varying_mean_permutation, δs, rej_rates_hipm, label = "hipm", color = "green", marker = (:circle, 4))
    plot!(varying_mean_permutation, δs, rej_rates_wow, label = "wow", color = "brown", marker = (:circle, 4))
    plot!(varying_mean_permutation, δs, rej_rates_energy, label = "Energy", color = "blue", marker = (:circle, 4))
    filepath = joinpath(pwd(), "frechet/figure1")
    savefig(varying_mean_permutation,joinpath(filepath, "varying_mean_n=$(n)_m=$(m)_S=$(S)_npermutation=$(n_permutation).png"))
end


function save_varying_variance_boostrap(n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)
    τs = collect(0.1:0.1:3.0)

    rej_rates_hipm = zeros(length(τs))
    rej_rates_wow = zeros(length(τs))
    rej_rates_dm = zeros(length(τs))
    rej_rates_energy = zeros(length(τs))

    for (i, τ) in enumerate(τs)
        μ_1, σ_1, a, b = 0.0, 0.2, -10.0, 10.0

        q_1 = tnormal_normal(μ_1, σ_1, a, b)
        q_2 = tnormal_normal(μ_1, σ_1*τ, a, b)

        rej_rates_hipm[i] = rejection_rate_hipm_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
        rej_rates_dm[i] = rejection_rate_dm_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
        rej_rates_wow[i] = rejection_rate_wow_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
        rej_rates_energy[i] = rejection_rate_energy_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
    end

    varying_variance_boostrap = plot(title = "Rejection rates of 4 testing schemes, boostrap", xlabel = "τ", ylabel = "Rej rate", 
                                        xlims=(0.0, 3.1), ylims = (-0.1, 1.1))
    plot!(varying_variance_boostrap, τs, rej_rates_dm, label = "dm", color = "red",marker = (:circle, 4))
    plot!(varying_variance_boostrap, τs, rej_rates_hipm, label = "hipm", color = "green", marker = (:circle, 4))
    plot!(varying_variance_boostrap, τs, rej_rates_wow, label = "wow", color = "brown", marker = (:circle, 4))
    plot!(varying_variance_boostrap, τs, rej_rates_energy, label = "Energy", color = "blue", marker = (:circle, 4))
    filepath = joinpath(pwd(), "frechet/figure1")
    savefig(varying_variance_boostrap,joinpath(filepath, "varying_variance_n=$(n)_m=$(m)_S=$(S)_nboostrap=$(n_boostrap).png"))
end

function save_varying_variance_permutation(n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)
    τs = collect(0.1:0.1:3.0)

    rej_rates_hipm = zeros(length(τs))
    rej_rates_wow = zeros(length(τs))
    rej_rates_dm = zeros(length(τs))
    rej_rates_energy = zeros(length(τs))

    for (i, τ) in enumerate(τs)
        μ_1, σ_1, a, b = 0.0, 0.2, -10.0, 10.0

        q_1 = tnormal_normal(μ_1, σ_1, a, b)
        q_2 = tnormal_normal(μ_1, σ_1*τ, a, b)

        rej_rates_hipm[i] = rejection_rate_hipm_permutation(q_1, q_2, n, m, S, θ, n_permutation)
        rej_rates_dm[i] = rejection_rate_dm_boostrap(q_1, q_2, n, m, S, θ, n_permutation)
        rej_rates_wow[i] = rejection_rate_wow_permutation(q_1, q_2, n, m, S, θ, n_permutation)
        rej_rates_energy[i] = rejection_rate_energy_boostrap(q_1, q_2, n, m, S, θ, n_permutation)
    end

    varying_variance_permutation = plot(title = "Rejection rates of 4 testing schemes, permutation", xlabel = "τ", ylabel = "Rej rate", 
                                        xlims=(0.0, 3.1), ylims = (-0.1, 1.1))
    plot!(varying_variance_permutation, τs, rej_rates_dm, label = "dm", color = "red",marker = (:circle, 4))
    plot!(varying_variance_permutation, τs, rej_rates_hipm, label = "hipm", color = "green", marker = (:circle, 4))
    plot!(varying_variance_permutation, τs, rej_rates_wow, label = "wow", color = "brown", marker = (:circle, 4))
    plot!(varying_variance_permutation, τs, rej_rates_energy, label = "Energy", color = "blue", marker = (:circle, 4))
    filepath = joinpath(pwd(), "frechet/figure1")
    savefig(varying_variance_permutation,joinpath(filepath, "varying_variance_n=$(n)_m=$(m)_S=$(S)_npermutation=$(n_permutation).png"))
end

function save_counterexample_boostrap(n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)

    λs = collect(0.0:0.1:1.0)

    rej_rates_hipm = zeros(length(λs))
    rej_rates_wow = zeros(length(λs))
    rej_rates_dm = zeros(length(λs))
    rej_rates_energy = zeros(length(λs))

    q_1 = simple_discr_1()
    q_2_aux = simple_discr_2()

    for (i, λ) in enumerate(λs)
        q_2 = mixture_ppm(q_1, q_2_aux, λ)

        rej_rates_hipm[i] = rejection_rate_hipm_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
        rej_rates_dm[i] = rejection_rate_dm_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
        rej_rates_wow[i] = rejection_rate_wow_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
        rej_rates_energy[i] = rejection_rate_energy_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
    end
    counterexmaple_boostrap = plot(title = "Rejection rates of 4 testing schemes, boostrap", xlabel = "λ", ylabel = "Rej rate", 
                                    xlims=(-0.1, 1.1), ylims = (-0.1, 1.1))
    plot!(counterexmaple_boostrap, λs, rej_rates_dm, label = "dm", color = "red",marker = (:circle, 4))
    plot!(counterexmaple_boostrap, λs, rej_rates_hipm, label = "hipm", color = "green", marker = (:circle, 4))
    plot!(counterexmaple_boostrap, λs, rej_rates_wow, label = "wow", color = "brown", marker = (:circle, 4))
    plot!(counterexmaple_boostrap, λs, rej_rates_energy, label = "Energy", color = "blue", marker = (:circle, 4))
    filepath = joinpath(pwd(), "frechet/counterexample")
    savefig(counterexmaple_boostrap,joinpath(filepath, "counterexample_boostrap_n=$(n)_m=$(m)_S=$(S)_nboostrap=$(n_boostrap).png"))
end


function save_counterexample_permutation(n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)

    λs = collect(0.0:0.1:1.0)

    rej_rates_hipm = zeros(length(λs))
    rej_rates_wow = zeros(length(λs))
    rej_rates_dm = zeros(length(λs))
    rej_rates_energy = zeros(length(λs))

    q_1 = simple_discr_1()
    q_2_aux = simple_discr_2()

    for (i, λ) in enumerate(λs)
        q_2 = mixture_ppm(q_1, q_2_aux, λ)

        rej_rates_hipm[i] = rejection_rate_hipm_permutation(q_1, q_2, n, m, S, θ, n_permutation)
        rej_rates_dm[i] = rejection_rate_dm_boostrap(q_1, q_2, n, m, S, θ, n_permutation)
        rej_rates_wow[i] = rejection_rate_wow_permutation(q_1, q_2, n, m, S, θ, n_permutation)
        rej_rates_energy[i] = rejection_rate_energy_boostrap(q_1, q_2, n, m, S, θ, n_permutation)
    end
    counterexmaple_permutation = plot(title = "Rejection rates of 4 testing schemes, permutation", xlabel = "λ", ylabel = "Rej rate", 
                                    xlims=(-0.1, 1.1), ylims = (-0.1, 1.1))
    plot!(counterexmaple_permutation, λs, rej_rates_dm, label = "dm", color = "red",marker = (:circle, 4))
    plot!(counterexmaple_permutation, λs, rej_rates_hipm, label = "hipm", color = "green", marker = (:circle, 4))
    plot!(counterexmaple_permutation, λs, rej_rates_wow, label = "wow", color = "brown", marker = (:circle, 4))
    plot!(counterexmaple_permutation, λs, rej_rates_energy, label = "Energy", color = "blue", marker = (:circle, 4))
    filepath = joinpath(pwd(), "frechet/counterexample")
    savefig(counterexmaple_permutation,joinpath(filepath, "counterexample_permutation_n=$(n)_m=$(m)_S=$(S)_npermutation=$(n_permutation).png"))
end





function save_counterexample_boostrap_only_dm(n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)

    λs = collect(0.0:0.1:1.0)

    rej_rates_dm = zeros(length(λs))
   

    q_1 = simple_discr_1()
    q_2_aux = simple_discr_2()

    for (i, λ) in enumerate(λs)
        q_2 = mixture_ppm(q_1, q_2_aux, λ)

        rej_rates_dm[i] = rejection_rate_dm_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
    end
    counterexmaple_boostrap = plot(title = "Rejection rate of dm", xlabel = "λ", ylabel = "Rej rate", 
                                    xlims=(-0.1, 1.1), ylims = (-0.1, 1.1))
    plot!(counterexmaple_boostrap, λs, rej_rates_dm, label = "dm", color = "red",marker = (:circle, 4))
    filepath = joinpath(pwd(), "frechet/counterexample")
    savefig(counterexmaple_boostrap,joinpath(filepath, "dm_countexample_n=$(n)_m=$(m)_S=$(S)_nboostrap=$(n_boostrap).png"))
end




# obtain times

q_1 = tnormal_normal(1.0, 1.0, -10.0, 10.0)
q_2 = tnormal_normal(1.0, 1.0, -10.0, 10.0)


n = 100
m = 100
S = 20
n_boostrap = 100
θ = 0.05
times = Dict()

t = time()
rej_rate = rejection_rate_dm_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
t = time() - t
times["dm"] = t / S

t = time()
rej_rate = rejection_rate_energy_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
t = time() - t
times["energy"] = t / S

t = time()
rej_rate = rejection_rate_hipm_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
t = time() - t
times["hipm"] = t / S
  
t = time()
rej_rate = rejection_rate_wow_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
t = time() - t
times["wow"] = t / S







# Reproduce figures from the paper Dubbey & Muller
# n, m = 5, 3
# S = 1
# θ = 0.05
# n_boostrap = 1
# n_permutation = n_boostrap

# t = time()
# save_varying_mean_boostrap(n,m,S,θ,n_boostrap)
# save_varying_mean_permutation(n,m,S,θ,n_permutation)

# save_varying_variance_boostrap(n,m,S,θ,n_boostrap)
# save_varying_variance_permutation(n,m,S,θ,n_permutation)
# duration = time() - t

# Example where method using Frechet mean and variance fails. 




# n = 100
# m = 100
# S = 500
# θ = 0.05
# n_boostrap = 100

# t = time()
# save_counterexample_boostrap_only_dm(n,m,S,θ,n_boostrap)
# duration = time() - t

# save_counterexample_boostrap(n,m,S,θ,n_boostrap)
# save_counterexample_permutation(n,m,S,θ,n_boostrap)







# t = time()
# q_1 = tnormal_normal(0.0, 0.5, -10.0, 10.0)
# q_2 = tnormal_normal(1.0, 0.5, -10.0, 10.0)
# rej_rate = rejection_rate_wow_boostrap(q_1, q_2, 100, 200, 4, 0.05, 2)
# duration = time() - t

# function dzveli(mu_1::Vector{Float64}, mu_2::Vector{Float64}, θ::Float64, n_boostrap::Int,)
#     @rput mu_1 mu_2 n_boostrap

#     R"""
#     # if (!requireNamespace("frechet", quietly = TRUE)) {
#     #   install.packages("frechet", repos="https://cloud.r-project.org")
#     # }

#     n1 <- length(mu_1)
#     n2 <- length(mu_2)
#     qSup <- seq(0.01, 0.99, length.out = 100)

#     Y1 <- lapply(seq_len(n1), function(i) qnorm(qSup, mean = mu_1[i], sd = 1.0))
#     Y2 <- lapply(seq_len(n2), function(i) qnorm(qSup, mean = mu_2[i], sd = 1.0))

#     Ly <- c(Y1, Y2)
#     Lx <- qSup
#     group <- c(rep(1, n1), rep(2, n2))

#     res <- frechet::DenANOVA(qin = Ly, supin = Lx, group = group,
#                     optns = list(boot = TRUE, R = n_boostrap))

#     p_boot <- res$pvalBoot
#     """
#     @rget p_boot
#     return 1 * (p_boot < θ)
# end

# function axali(atoms_1::Matrix{Float64}, atoms_2::Matrix{Float64}, θ::Float64, n_boostrap::Int)

#     rej_rate = 0.0
#     n = size(atoms_1)[1]
    
#     @rput atoms_1 atoms_2 n n_boostrap
#     R"""
#     # if (!requireNamespace("frechet", quietly = TRUE)) {
#     #   install.packages("frechet", repos="https://cloud.r-project.org")
#     # }
    
#     atoms_all = rbind(atoms_1, atoms_2)

#     group <- c(rep(1, n), rep(2, n))

#     result_denanova = frechet::DenANOVA(
#         yin = atoms_all,
#         group = group,
#         optns = list(boot = TRUE, R = n_boostrap)
#     )
#     pvalue = result_denanova$pvalBoot
#     """
#     @rget pvalue
#     rej_rate += 1 * (pvalue < θ)

#     return rej_rate
# end



# S = 10
# n = 100
# m = 200
# n_boostrap = 100
# decisions_dzveli = 0.0
# decisions_axali = 0.0
# t = time()
# for s in 1:S
#     atoms_1 = zeros(n,m)
#     atoms_2 = zeros(n,m)
#     mu_1 = rand(Normal(0.0,1.0), n)
#     mu_2 = rand(Normal(0.0,1.1), n)
#     for i in 1:n
#         atoms_1[i,:] = rand(Normal(mu_1[i]),m)
#         atoms_2[i,:] = rand(Normal(mu_2[i]),m)
    
#     end
#     global decisions_axali += axali(atoms_1,atoms_2,θ,n_boostrap)
#     #global decisions_dzveli += dzveli(mu_1, mu_2, θ, n_boostrap,)
# end

# duration = time() - t
# decisions_dzveli /= S
# decisions_axali /= S
    

# 90 seconds for S - 100, boost = 100
# 0.2 seconds for S = 1, boost = 1.