# This file contains functions for simulations: test statistics, decision of rejection, thresholds and rejection rates.


using RCall # to call R functions
using FLoops # for parallel computing

include("distributions.jl")

include("distances/hipm.jl")
include("distances/wow.jl") 


"""
    test_statistic_energy

Given two matrices of atoms (hierarchical sample using definition in paper), computes the test statitic from Szekely et al. (2004).

# Arguments:
    atoms_1::AbstractArray{Float64, 2}
    atoms_2::AbstractArray{Float64, 2}

# Warning: 
    Each row in matrices atoms_1/2 must be sorted.
    Number of rows of atoms_1 and atoms_2 should be the same.
"""

function test_statistic_energy(atoms_1::AbstractArray{Float64, 2}, atoms_2::AbstractArray{Float64, 2})
    
    n = size(atoms_1)[1]
    n == size(atoms_2)[1] || throw(ArgumentError("Number of rows of atoms_1 and atoms_2 are not the same."))
    
    sum_distances_x = 0.0 # collects sum of all possible distances in atoms_1
    sum_distances_xy = 0.0 # collects sum of all possible distances between atoms_1 and atoms_2
    sum_distances_y = 0.0 # collects sum of all possible distances in atoms_2

    for i in 1:n
        x = @view atoms_1[i,:]
        y = @view atoms_2[i,:]
        for j in 1:n
            x_j = @view atoms_1[j,:]
            y_j = @view atoms_2[j,:]

            sum_distances_x += wasserstein_1d_equal(x, x_j)
            sum_distances_xy += wasserstein_1d_equal(x, y_j)
            sum_distances_y += wasserstein_1d_equal(y, y_j)
        end
    end
    distance = 2 * sum_distances_xy / (n * n) - sum_distances_x / (n * n) -sum_distances_y / (n * n)
    return distance * n / 2
end


"""
    decision_energy

Given two hierarchical samples, decides whether to reject null hypothesis using the method from Szekely et al. (2004).

# Arguments: 
    h_1::HierSample
    h_2::HierSample
    θ::Float64       :  Significance level
    n_samples::Int   :  number samples for Bootstrap approach
"""

function decision_energy(h_1::HierSample, h_2::HierSample, θ::Float64, n_samples::Int)

    atoms_1 = h_1.atoms
    atoms_2 = h_2.atoms
    n = size(atoms_1)[1]

    observed_test_stat = test_statistic_energy(atoms_1, atoms_2)
        
    # obtain quantile using bootstrap approach
    bootstrap_samples = zeros(n_samples) 
    pooled_atoms = vcat(atoms_1, atoms_2) # collect all rows
    for i in 1:n_samples
        indices_1 = sample(1:2*n, n; replace = true)
        indices_2 = sample(1:2*n, n; replace = true)

        new_atoms_1 = @view pooled_atoms[indices_1,:]
        new_atoms_2 = @view pooled_atoms[indices_2,:]
        bootstrap_samples[i] = test_statistic_energy(new_atoms_1, new_atoms_2)
    end
    pvalue = mean(bootstrap_samples .>= observed_test_stat)
    return Float64(pvalue < θ)
end


"""
    decision_dm

Given two vectors, each containing the means of Gaussian distributions with variance 1, decides whether to reject H_0
according to the method from Dubey and Muller (2019).

# Arguments:
    means_1::Vector{Float64}  :  means of Gaussians with variance 1
    means_2::Vector{Float64}  :  means of Gaussians with variance 1
    θ::Float64                :  Significance level
    n_bootstrap::Int          :  number of bootstrap samples.
"""
function decision_dm(means_1::Vector{Float64}, means_2::Vector{Float64}, θ::Float64, n_bootstrap::Int)
  
    n = length(means_1)
    
    @rput means_1 means_2 n n_bootstrap
    R"""

    library(frechet)
    n_1 <- n
    n_2 <- n
    delta <- 1
    qSup <- seq(0.01, 0.99, (0.99 - 0.01) / 50)

    Y1 <- lapply(1:n_1, function(i) {
    qnorm(qSup, means_1[i], sd = 1)
    })
    Y2 <- lapply(1:n_2, function(i) {
    qnorm(qSup, means_2[i], sd = 1)
    })
    Ly <- c(Y1, Y2)
    Lx <- qSup
    group <- c(rep(1, n_1), rep(2, n_2))
    res <- DenANOVA(qin = Ly, supin = Lx, group = group, optns = list(boot = TRUE, R = n_bootstrap))
    pvalue = res$pvalBoot 
    """
    @rget pvalue  
    return Float64(pvalue < θ)
end


"""
    decision_dm

Given hierarchical sample objects, decides whether to reject H_0 according to the method from Dubey and Muller (2019).

# Arguments:
    h_1::HierSample
    h_2::HierSample
    θ::Float64                :  Significance level
    n_bootstrap::Int          :  number of bootstrap samples.

# Warning:
    if in each row, most of the atoms are repeating, may give an error.
"""
function decision_dm(h_1::HierSample, h_2::HierSample, θ::Float64, n_bootstrap::Int)

    n_1 = size(h_1.atoms)[1]
    n_2 = size(h_2.atoms)[1]

    pooled_atoms = vcat(h_1.atoms, h_2.atoms)
    
    @rput pooled_atoms n_1 n_2 n_bootstrap
    R"""

    library(frechet)
    
    group <- c(rep(1, n_1), rep(2, n_2))
    res <- DenANOVA(yin = pooled_atoms, group = group, optns = list(boot = TRUE, R = n_bootstrap))
    pvalue = res$pvalBoot # returns bootstrap pvalue
    """
    @rget pvalue  
    return Float64(pvalue < θ)
end



"""
    rejection_rate_dm

Given specific laws of RPMS (laws on N(δ, 1)), estimates the rejection rate of the testing scheme
according to Dubey and Muller (2019). It works directly on probability measures instead of hierarchical samples.

# Arguments:
    q_1::Union{tnormal_normal, discr_normal, mixture}  :  law of RPM
    q_2::Union{tnormal_normal, discr_normal, mixture}  :  law of RPM
    n::Int                                                 :  number of probability measures
    S::Int                                                 :  number of MCMC iterations to estimate rejection rate
    θ::Float64                                             :  significance level
    n_bootstrap::Int                                       :  number of bootstrap samples.
"""

function rejection_rate_dm(q_1::Union{tnormal_normal, discr_normal, mixture},
                             q_2::Union{tnormal_normal, discr_normal, mixture}, 
                             n::Int,S::Int, θ::Float64, n_bootstrap::Int)
    rej_rate = 0.0
    
    for i in 1:S
        # generate normal distributions 
        means_1 = generate_prob_measures(q_1, n) # only contains means for normal distribution
        means_2 = generate_prob_measures(q_2, n) # only contains means for normal distribution
        
        rej_rate += decision_dm(means_1, means_2, θ, n_bootstrap) 
    end
    return rej_rate/S
end


"""
    threshold_hipm

Given hierchical samples, obtains the threshold using HIPM via bootstrap/permutation approach.

# Arguments: 
    h_1::HierSample
    h_2::HierSample
    θ::Union{Float64, Vector{Float64}}  :  Significance level/s
    n_samples::Int                      :  number of bootstrap/permutation samples
    bootstrap::Bool                     :  Boolean variable. If true use bootstrap, otherwise permutation.

"""
function threshold_hipm(h_1::HierSample, h_2::HierSample, θ::Union{Float64, Vector{Float64}}, n_samples::Int, bootstrap::Bool)
    # Obtains threshold for HIPM via permutation or bootstrap approach.
    n_1 = size(h_1.atoms)[1]
    n_2 = size(h_2.atoms)[1]
    n_total = n_1 + n_2

    a = minimum((h_1.a,h_2.a))
    b = maximum((h_1.b,h_2.b))
   
    samples = zeros(n_samples) # storing bootstrap/permutation samples
    pooled_atoms = vcat(h_1.atoms, h_2.atoms) # collect all rows

    if bootstrap
        for i in 1:n_samples
            indices_1 = sample(1:n_total, n_1; replace = true)
            indices_2 = sample(1:n_total, n_2; replace = true)

            new_atoms_1 = @view pooled_atoms[indices_1,:] 
            new_atoms_2 = @view pooled_atoms[indices_2,:] 

            samples[i] = dlip(new_atoms_1, new_atoms_2, a, b)
        end
    else
        for i in 1:n_samples
            random_indices = randperm(n_total) # indices to distribute rows to new hierarchical meausures

            new_atoms_1 = @view pooled_atoms[random_indices[1:n_1],:] # rows indexed by first n_1 random indices to the atoms_1
            new_atoms_2 = @view pooled_atoms[random_indices[n_1+1:end],:] # rows indexed by the rest of random indices to the atoms_2
        
            samples[i] = dlip(new_atoms_1, new_atoms_2, a, b)
        end
    end
    return quantile(samples, 1 .- θ) .* sqrt(n_1*n_2 / n_total)
end

"""
    decision_hipm

Given two hierarchical samples, decides whether to reject H_0 using HIPM via bootstrap/permutation approach,

# Arguments:
    h_1::HierSample
    h_2::HierSample
    θ::Float64       :  Significance level/s
    n_samples::Int   :  number of bootstrap/permutation samples
    bootstrap::Bool  :  Boolean variable. If true use bootstrap, otherwise permutation.
"""
function decision_hipm(h_1::HierSample, h_2::HierSample, θ::Union{Float64, Vector{Float64}}, n_samples::Int, bootstrap::Bool)
    
    atoms_1 = h_1.atoms
    atoms_2 = h_2.atoms
    n_1 = size(atoms_1)[1]
    n_2 = size(atoms_2)[1]
    n_total = n_1 + n_2

    a = minimum((h_1.a, h_2.a))
    b = maximum((h_1.b, h_2.b))
    observed_distance = dlip(h_1, h_2, a, b)
       
    # obtain bootstrap/permutation samples
    samples = zeros(n_samples) 
    pooled_atoms = vcat(atoms_1, atoms_2) # collect all rows

    if bootstrap
        for i in 1:n_samples
            indices_1 = sample(1:n_total, n_1; replace = true)
            indices_2 = sample(1:n_total, n_2; replace = true)

            new_atoms_1 = @view pooled_atoms[indices_1,:] 
            new_atoms_2 = @view pooled_atoms[indices_2,:] 

            samples[i] = dlip(new_atoms_1, new_atoms_2, a, b)
        end
    else
        for i in 1:n_samples
            random_indices = randperm(n_total) # indices to distribute rows to new hierarchical meausures

            new_atoms_1 = @view pooled_atoms[random_indices[1:n_1],:] # rows indexed by first n_1 random indices to the atoms_1
            new_atoms_2 = @view pooled_atoms[random_indices[n_1+1:end],:] # rows indexed by the rest of random indices to the atoms_2
        
            samples[i] = dlip(new_atoms_1, new_atoms_2, a, b)
        end
    end
    pvalue = mean(samples .>= observed_distance)
    return Float64.(pvalue .< θ)
end

"""
    threshold_wow

Given hierchical samples, obtains the threshold using WoW via bootstrap/permutation approach.

# Arguments: 
    h_1::HierSample
    h_2::HierSample
    θ::Union{Float64, Vector{Float64}}  :  Significance level/s
    n_samples::Int                      :  number of bootstrap/permutation samples
    bootstrap::Bool                     :  Boolean variable. If true use bootstrap, otherwise permutation.

"""
function threshold_wow(h_1::HierSample, h_2::HierSample, θ::Union{Float64, Vector{Float64}}, n_samples::Int, bootstrap::Bool)
    # Obtains threshold for HIPM via permutation or bootstrap approach.
    n_1 = size(h_1.atoms)[1]
    n_2 = size(h_2.atoms)[1]
    n_total = n_1 + n_2

    samples = zeros(n_samples) # storing bootstrap/permutation samples
    pooled_atoms = vcat(h_1.atoms, h_2.atoms) # collect all rows

    if bootstrap
        for i in 1:n_samples
            indices_1 = sample(1:n_total, n_1; replace = true)
            indices_2 = sample(1:n_total, n_2; replace = true)

            new_atoms_1 = @view pooled_atoms[indices_1,:] 
            new_atoms_2 = @view pooled_atoms[indices_2,:] 

            samples[i] = ww(new_atoms_1, new_atoms_2)
        end
    else
        for i in 1:n_samples
            random_indices = randperm(n_total) # indices to distribute rows to new hierarchical meausures

            new_atoms_1 = @view pooled_atoms[random_indices[1:n_1],:] # rows indexed by first n_1 random indices to the atoms_1
            new_atoms_2 = @view pooled_atoms[random_indices[n_1+1:end],:] # rows indexed by the rest of random indices to the atoms_2
        
            samples[i] = ww(new_atoms_1, new_atoms_2)
        end
    end
    return quantile(samples, 1 .- θ) .* sqrt(n_1*n_2 / n_total)
end


"""
    decision_wow

Given two hierarchical samples, decides whether to reject H_0 using WoW via bootstrap/permutation approach,

# Arguments:
    h_1::HierSample
    h_2::HierSample
    θ::Float64       :  Significance level/s
    n_samples::Int   :  number of bootstrap/permutation samples
    bootstrap::Bool  :  Boolean variable. If true use bootstrap, otherwise permutation.
"""
function decision_wow(h_1::HierSample, h_2::HierSample, θ::Union{Float64, Vector{Float64}}, n_samples::Int, bootstrap::Bool)
    
    atoms_1 = h_1.atoms
    atoms_2 = h_2.atoms
    n_1 = size(atoms_1)[1]
    n_2 = size(atoms_2)[1]
    n_total = n_1 + n_2

  
    observed_distance = ww(h_1, h_2)
       
    # obtain bootstrap/permutation samples
    samples = zeros(n_samples) 
    pooled_atoms = vcat(atoms_1, atoms_2) # collect all rows

    if bootstrap
        for i in 1:n_samples
            indices_1 = sample(1:n_total, n_1; replace = true)
            indices_2 = sample(1:n_total, n_2; replace = true)

            new_atoms_1 = @view pooled_atoms[indices_1,:] 
            new_atoms_2 = @view pooled_atoms[indices_2,:] 

            samples[i] = ww(new_atoms_1, new_atoms_2)
        end
    else
        for i in 1:n_samples
            random_indices = randperm(n_total) # indices to distribute rows to new hierarchical meausures

            new_atoms_1 = @view pooled_atoms[random_indices[1:n_1],:] # rows indexed by first n_1 random indices to the atoms_1
            new_atoms_2 = @view pooled_atoms[random_indices[n_1+1:end],:] # rows indexed by the rest of random indices to the atoms_2
        
            samples[i] = ww(new_atoms_1, new_atoms_2)
        end
    end
    pvalue = mean(samples .>= observed_distance)
    return Float64.(pvalue .< θ)
end


"""
    rejection_rate_hipm_wow

Given two laws of RPMs, returns rejection rates using HIPM and WoW.

# Arguments: 
    q_1::LawRPM
    q_2::LawRPM 
    n::Int              :  number of exchangeable sequences
    m::Int              :  length of each exchangeable sequence 
    S::Int              :  number of MCMC samples to estimate rejection rate 
    θ::Vector{Float64}  :  siginicance level/s
    n_samples::Int      :  number of bootstrap/permutation samples
    bootstrap::Bool     :  Boolean variable, if true use bootstrap approach, otherwise permutation.
"""
function rejection_rate_hipm_wow(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, S::Int, θ::Vector{Float64},
                     n_samples::Int, bootstrap::Bool)
    rates_hipm = zeros(length(θ))
    rates_wow = zeros(length(θ))
    @floop ThreadedEx() for s in 1:S
        # generate samples
        h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)

        # record decisions from each testing methods
        @reduce rates_hipm .+= decision_hipm(h_1, h_2, θ, n_samples, bootstrap)
        @reduce rates_wow .+= decision_wow(h_1, h_2, θ, n_samples, bootstrap)
    end
    rates_hipm ./= S
    rates_wow ./= S
    
    if length(θ) == 1
        return rates_hipm[1],rates_wow[1]
    else
        return rates_hipm, rates_wow
    end
end




"""
    rejection_rate_all

Given two laws of RPMs, returns rejection rates for all 4 testing schemes: HIPM, WoW, DM, Energy. Note that for HIPM, 
WoW and Energy test, we record decisions on same hierarchical samples; On the other hand, we record seperately decisions for DM. 

# Arguments: 
    q_1::LawRPM
    q_2::LawRPM 
    n::Int              :  number of exchangeable sequences
    m::Int              :  length of each exchangeable sequence 
    S::Int              :  number of MCMC samples to estimate rejection rate 
    θ::Float64          :  siginicance level
    n_samples::Int      :  number of bootstrap/permutation samples
    bootstrap::Bool     :  Boolean variable, if true use bootstrap approach, otherwise permutation.
"""
function rejection_rate_all(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, S::Int, θ::Float64, n_samples::Int, bootstrap::Bool)
    rates_hipm = 0.0
    rates_wow = 0.0
    rates_energy = 0.0

    @floop ThreadedEx() for s in 1:S
        # generate samples and set endpoints
        h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)

        # record decisions from each testing methods
        @reduce rates_hipm += decision_hipm(h_1, h_2, θ, n_samples, bootstrap)
        @reduce rates_wow += decision_wow(h_1, h_2, θ, n_samples, bootstrap)
        @reduce rates_energy += decision_energy(h_1, h_2, θ, n_samples) 
    end
    rates_energy /= S
    rates_wow /= S
    rates_hipm /= S
    rates_dm = rejection_rate_dm(q_1, q_2, n, S, θ, n_samples)
    
    return rates_hipm,rates_wow,rates_dm,rates_energy
end



"""

pvalue_hipm

This function is specifically for mortality dataset. If pool is false, given atoms and weights associated to two Hierarchical estimators,
we estimate the p-value for HIPM via permutation approach. otherwise, we pool the probability measures (average inside group) and then get the
p-values using Wasserstein distance between them.

# Arguments:
    atoms_1::Matrix{Float64}
    weights_1::Matrix{Float64}
    atoms_2::Matrix{Float64}
    weights_2::Matrix{Float64}
    n_permutations::Int
    max_time::Float  :  number of seconds for runnnig optimizaiton algorithm in HIPM
    pooled::Bool  :  Boolean variable to denote whether to pool the probability measures in the group.
"""

function pvalue_hipm(atoms_1::Matrix{Float64}, weights_1::Matrix{Float64}, atoms_2::Matrix{Float64}, weights_2::Matrix{Float64},
     n_permutations::Int, max_time::Float64, pooled::Bool)
    n_1 = size(atoms_1,1)
    n_2 = size(atoms_2,1)
    n = n_1 + n_2
    a = atoms_1[1,1]
    b = atoms_1[1,end]
    if pooled
        pooled_weights_1 = vec(mean(weights_1, dims = 1))
        pooled_weights_2 = vec(mean(weights_2, dims = 1))
        cdf_diff =abs.(cumsum(pooled_weights_1) .- cumsum(pooled_weights_2))
        T_observed = sum(cdf_diff)
    else
        T_observed = dlip(atoms_1,atoms_2, weights_1, weights_2, a, b; max_time = max_time)
    end
    samples = zeros(n_permutations)
    total_weights = vcat(weights_1, weights_2) # collect all rows

    if pooled
        for i in 1:n_permutations
            random_indices = randperm(n) # indices to distribute rows to new hierarchical meausures

            new_weights_1 = total_weights[random_indices[1:n_1],:] # first rows indexed by n random indices to the atoms_1
            new_weights_2 = total_weights[random_indices[n_1+1:end],:] # first rows indexed by n random indices to the atoms_2
            pooled_weights_1 = vec(mean(new_weights_1, dims = 1))
            pooled_weights_2 = vec(mean(new_weights_2, dims = 1))
            cdf_diff =abs.(cumsum(pooled_weights_1) .- cumsum(pooled_weights_2))
            samples[i] = sum(cdf_diff)
        end
    else
        @floop ThreadedEx() for i in 1:n_permutations
            random_indices = randperm(n) # indices to distribute rows to new hierarchical meausures

            new_weights_1 = total_weights[random_indices[1:n_1],:] # first rows indexed by n random indices to the atoms_1
            new_weights_2 = total_weights[random_indices[n_1+1:end],:] # first rows indexed by n random indices to the atoms_2

            samples[i] = dlip(atoms_1, atoms_2, new_weights_1, new_weights_2, a, b; max_time = max_time)
        end 
    end
    return mean(samples.>=T_observed)
end 













# up to now more or less everything is fine



# function rejection_rate_all_fake(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, S::Int, θ::Float64, n_samples::Int, bootstrap::Bool)
#     # Given two laws of RPMs, returns rejection rate for all 4 testing schemes.
    
#     # It is called fake because thresholds for hipm and wow are obtained from some auxiliary hierarchical
#     # samples and then used for each simulated sample.

#     # firstly we obtain fixed thresholds for HIPM and WoW
#     aux_h_1 = generate_hiersample(q_1,n,m)
#     aux_h_2 = generate_hiersample(q_2, n, m)
#     threshold_hipm_wrong = threshold_hipm(aux_h_1, aux_h_2, θ, n_samples, bootstrap) # gasaketebeli
#     threshold_wow_wrong = threshold_wow(aux_h_1, aux_h_2, θ, n_samples, bootstrap) # gasaketebeli

#     rates_hipm = 0.0
#     rates_wow = 0.0
#     rates_energy = 0.0

#     @floop ThreadedEx() for s in 1:S
#         # generate samples and set endpoints
#         h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
#         a = minimum((h_1.a, h_2.a))
#         b = maximum((h_1.b, h_2.b))
#         h_1.a = a
#         h_2.a = a
#         h_1.b = b
#         h_2.b = b

#         # record decisions from each testing methods
#         @reduce rates_hipm += 1.0*(dlip(h_1, h_2) > threshold_hipm_wrong)
#         @reduce rates_wow += 1.0 * (ww(h_1, h_2) > threshold_wow_wrong)
#         @reduce rates_energy += decision_energy(h_1, h_2, θ, n_samples) 
#     end
#     rates_energy /= S
#     rates_wow /= S
#     rates_hipm /= S
#     rates_dm = 0.0
#     rates_dm = rejection_rate_dm(q_1, q_2, n, S, θ, n_samples)
#     return rates_hipm,rates_wow,rates_dm,rates_energy
# end






# function rejection_rate_wow(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, S::Int,
#                      threshold_wow_wrong::Float64)
#     # if bootstrap is true then do bootstrap approach, n_samples refers to either number of permutations or bootstraps



#     rates_wow = 0.0
  
#     @floop ThreadedEx() for s in 1:S
#         # generate samples and set endpoints
#         h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
        

#         # record decisions from each testing methods
#         @reduce rates_wow += 1.0 * (ww(h_1, h_2) > threshold_wow_wrong)
#     end
#     rates_wow /= S
#     return rates_wow
# end

# function rejection_rate_wow(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, S::Int,
#                      θ::Float64, n_samples::Int, bootstrap::Bool)
#     # firstly obtain threshold
#     aux_h_1 = generate_hiersample(q_1,n,m)
#     aux_h_2 = generate_hiersample(q_2, n, m)
#     threshold_wow_wrong = threshold_wow(aux_h_1, aux_h_2, θ, n_samples, bootstrap) # gasaketebeli
#     return rejection_rate_wow(q_1, q_2, n, m, S, threshold_wow_wrong)
# end










# function rejection_rate_hipm_wow(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, S::Int, θ::Float64, n_samples::Int, bootstrap::Bool)
#     # if bootstrap is true then do bootstrap approach, n_samples refers to either number of permutations or bootstraps

#     # firstly we obtain fixed thresholds for HIPM and WoW
#     aux_h_1 = generate_hiersample(q_1,n,m)
#     aux_h_2 = generate_hiersample(q_2, n, m)
#     threshold_hipm_wrong = threshold_hipm(aux_h_1, aux_h_2, θ, n_samples, bootstrap) # gasaketebeli
#     threshold_wow_wrong = threshold_wow(aux_h_1, aux_h_2, θ, n_samples, bootstrap) # gasaketebeli

#     rates_hipm = 0.0
#     rates_wow = 0.0

#     @floop ThreadedEx() for s in 1:S
#         # generate samples and set endpoints
#         h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
#         a = minimum((h_1.a, h_2.a))
#         b = maximum((h_1.b, h_2.b))
#         h_1.a = a
#         h_2.a = a
#         h_1.b = b
#         h_2.b = b

#         # record decisions from each testing methods
#         @reduce rates_hipm += 1.0*(dlip(h_1, h_2) > threshold_hipm_wrong)
#         @reduce rates_wow += 1.0 * (ww(h_1, h_2) > threshold_wow_wrong)
#     end
#     rates_wow /= S
#     rates_hipm /= S
#     return rates_hipm,rates_wow
# end



# function save_fig_hipm_wow(pairs::Vector{<:Tuple{LawRPM,LawRPM}}, param_pairs::Vector{Float64}, file_name::String, file_path::String, title::String, xlabel::String, ylabel::String,
#     n::Int, m::Int, S::Int, θ::Float64, n_samples::Int, bootstrap::Bool)
#     rates_hipm = zeros(length(param_pairs))
#     rates_wow = zeros(length(param_pairs))
#     for i in 1:length(pairs)
#         q_1, q_2 = pairs[i]
#         r_hipm, r_wow = rejection_rate_hipm_wow(q_1,q_2,n,m,S,θ,n_samples,bootstrap)
#         rates_hipm[i] = r_hipm
#         rates_wow[i] = r_wow
#         println(i)
#     end
#     fig = plot(title = title, xlabel = xlabel, ylabel = ylabel, xlims=(minimum(param_pairs) - 0.10, maximum(param_pairs)+ 0.10),
#                          ylims = (-0.1, 1.1))
#     plot!(fig, param_pairs, rates_hipm, label = "hipm", color = "green", marker = (:circle, 4))
#     plot!(fig, param_pairs, rates_wow, label = "wow", color = "brown", marker = (:circle, 4))
#     filepath = joinpath(pwd(), file_path)
#     savefig(fig,joinpath(filepath, file_name))
# end





# function rejection_rate_energy_boostrap_parallel(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)
#     rej_rate = 0.0

#     @floop ThreadedEx() for s in 1:S
#         h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
#         observed_test_stat = test_statistic_energy(h_1, h_2)
        
#         # obtain quantile using bootstrap approach
#         boostrap_samples = zeros(n_boostrap) # zeros can be improved
#         a = minimum([h_1.a, h_2.a])
#         b = maximum([h_1.b, h_2.b])
#         pooled_atoms = vcat(h_1.atoms, h_2.atoms) # collect all rows
#         for i in 1:n_boostrap
#             indices_1 = sample(1:2*n, n; replace = true)
#             indices_2 = sample(1:2*n, n; replace = true)
#             atoms_1 = pooled_atoms[indices_1,:]  # resample from pooled hierarchical sample
#             atoms_2 = pooled_atoms[indices_2,:]  # resample from pooled hierarchical sample
            
        
#             h_1_boostrap = HierSample(atoms_1, n, m, a, b)
#             h_2_boostrap = HierSample(atoms_2, n, m, a, b)

#             boostrap_samples[i] = test_statistic_energy(h_1_boostrap, h_2_boostrap)
#         end
#         threshold = quantile(boostrap_samples, 1 - θ)
        
#         @reduce rej_rate += Float64(observed_test_stat > threshold)

#     end
#     return rej_rate / S
# end



# function decision_dm(h_1::HierSample, h_2::HierSample, θ::Float64, n_samples::Int)
#     atoms_1 = copy(h_1.atoms)
#     atoms_2 = copy(h_2.atoms)
#     n = size(h_1.atoms)[1]     
    
#     @rput atoms_1 atoms_2 n n_samples
#     R"""
#     # if (!requireNamespace("frechet", quietly = TRUE)) {
#     #   install.packages("frechet", repos="https://cloud.r-project.org")
#     # }
    
#     library(frechet)
#     atoms_all = rbind(atoms_1, atoms_2)

#     group <- c(rep(1, n), rep(2, n))

#     result_denanova = DenANOVA(
#         yin = atoms_all,
#         group = group,
#         optns = list(boot = TRUE, R = n_samples)
#     )
#     pvalue = result_denanova$pvalBoot
#     """
#     @rget pvalue
#     return Float64(pvalue < θ)
# end



# function rejection_rate_dm_boostrap(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)

#     rej_rate = 0.0
#     for s in 1:S
     
#         h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
#         atoms_1, atoms_2 = h_1.atoms, h_2.atoms

#         @rput atoms_1 atoms_2 n n_boostrap
#         R"""
#         # if (!requireNamespace("frechet", quietly = TRUE)) {
#         #   install.packages("frechet", repos="https://cloud.r-project.org")
#         # }
#         library(frechet)
#         atoms_all = rbind(atoms_1, atoms_2)

#         group <- c(rep(1, n), rep(2, n))

#         result_denanova = DenANOVA(
#             yin = atoms_all,
#             group = group,
#             optns = list(boot = TRUE, R = n_boostrap)
#         )
#         pvalue = result_denanova$pvalBoot
#         """
#         @rget pvalue
#         rej_rate += Float64(pvalue < θ)
#     end
#     rej_rate /= S
#     return rej_rate
# end



# function rejection_rate_hipm_boostrap_parallel(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)
#     rej_rate = 0.0

#     @floop ThreadedEx() for s in 1:S
#         h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
#         a = minimum([h_1.a, h_2.a])
#         b = maximum([h_1.b, h_2.b])
#         h_1.a = a
#         h_2.a = a
#         h_1.b = b
#         h_2.b = b
#         observed_test_stat = dlip(h_1, h_2)
        
#         # obtain quantile using bootstrap approach
#         boostrap_samples = zeros(n_boostrap) # zeros can be improved
        
#         pooled_atoms = vcat(h_1.atoms, h_2.atoms) # collect all rows
#         for i in 1:n_boostrap
#             indices_1 = sample(1:2*n, n; replace = true)
#             indices_2 = sample(1:2*n, n; replace = true)
#             atoms_1 = pooled_atoms[indices_1,:]  # resample from pooled hierarchical sample
#             atoms_2 = pooled_atoms[indices_2,:]  # resample from pooled hierarchical sample
            
        
#             h_1_boostrap = HierSample(atoms_1, n, m, a, b)
#             h_2_boostrap = HierSample(atoms_2, n, m, a, b)

#             boostrap_samples[i] = dlip(h_1_boostrap, h_2_boostrap)
#         end
#         threshold = quantile(boostrap_samples, 1 - θ)
        
#         @reduce rej_rate += Float64(observed_test_stat > threshold)

#     end
#     return rej_rate / S
# end


# function rejection_rate_hipm_permutation_wrong(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)

#     # firstly we obtain threshold
#     h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
#     a = minimum([h_1.a, h_2.a])
#     b = maximum([h_1.b, h_2.b])
#     h_1.a = a
#     h_2.a = a
#     h_1.b = b
#     h_2.b = b

#     permutation_samples = zeros(n_permutation) # zeros can be improved
#     pooled_atoms = vcat(h_1.atoms, h_2.atoms) # collect all rows
#         for i in 1:n_permutation
#             random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

#             atoms_1 = pooled_atoms[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
#             atoms_2 = pooled_atoms[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
#             h_1_permutation = HierSample(atoms_1, n, m, a, b)
#             h_2_permutation = HierSample(atoms_2, n, m, a, b)

#             permutation_samples[i] = dlip(h_1_permutation, h_2_permutation)
#         end
#     threshold = quantile(permutation_samples, 1 - θ)

#     rej_rate = 0.0

#     @floop ThreadedEx() for s in 1:S
#         local h_1 = generate_hiersample(q_1, n, m)
#         local h_2 = generate_hiersample(q_2, n, m)
#         local a = minimum([h_1.a, h_2.a])
#         local b = maximum([h_1.b, h_2.b])
#         h_1.a = a
#         h_2.a = a
#         h_1.b = b
#         h_2.b = b
#         observed_test_stat = dlip(h_1, h_2)
        
#         @reduce rej_rate += Float64(observed_test_stat > threshold)

#     end
#     return rej_rate / S
# end



# function rejection_rate_hipm_permutation_parallel(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)
#     rej_rate = 0.0

#     @floop ThreadedEx() for s in 1:S
#         h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
#         a = minimum([h_1.a, h_2.a])
#         b = maximum([h_1.b, h_2.b])
#         h_1.a = a
#         h_2.a = a
#         h_1.b = b
#         h_2.b = b
#         observed_test_stat = dlip(h_1, h_2)
        
#         # obtain quantile using permutation approach
#         permutation_samples = zeros(n_permutation) # zeros can be improved
        
#         pooled_atoms = vcat(h_1.atoms, h_2.atoms) # collect all rows
#         for i in 1:n_permutation
#             random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

#             atoms_1 = pooled_atoms[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
#             atoms_2 = pooled_atoms[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
#             h_1_permutation = HierSample(atoms_1, n, m, a, b)
#             h_2_permutation = HierSample(atoms_2, n, m, a, b)

#             permutation_samples[i] = dlip(h_1_permutation, h_2_permutation)
#         end
#         threshold = quantile(permutation_samples, 1 - θ)
        
#         @reduce rej_rate += Float64(observed_test_stat > threshold)

#     end
#     return rej_rate / S
# end



# function rejection_rate_wow_boostrap_parallel(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)
#     rej_rate = 0.0

#     @floop ThreadedEx() for s in 1:S
#         h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
#         a = minimum([h_1.a, h_2.a])
#         b = maximum([h_1.b, h_2.b])
#         h_1.a = a
#         h_2.a = a
#         h_1.b = b
#         h_2.b = b
#         observed_test_stat = ww(h_1, h_2)
        
#         # obtain quantile using bootstrap approach
#         boostrap_samples = zeros(n_boostrap) # zeros can be improved
  
#         pooled_atoms = vcat(h_1.atoms, h_2.atoms) # collect all rows
#         for i in 1:n_boostrap
#             indices_1 = sample(1:2*n, n; replace = true)
#             indices_2 = sample(1:2*n, n; replace = true)
#             atoms_1 = pooled_atoms[indices_1,:]  # resample from pooled hierarchical sample
#             atoms_2 = pooled_atoms[indices_2,:]  # resample from pooled hierarchical sample
            
        
#             h_1_boostrap = HierSample(atoms_1, n, m, a, b)
#             h_2_boostrap = HierSample(atoms_2, n, m, a, b)

#             boostrap_samples[i] = ww(h_1_boostrap, h_2_boostrap)
#         end
#         threshold = quantile(boostrap_samples, 1 - θ)
        
#         @reduce rej_rate += Float64(observed_test_stat > threshold)

#     end
#     return rej_rate / S
# end


# function rejection_rate_wow_permutation_wrong(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)

#     # firstly we obtain threshold
#     h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
#     a = minimum([h_1.a, h_2.a])
#     b = maximum([h_1.b, h_2.b])
#     h_1.a = a
#     h_2.a = a
#     h_1.b = b
#     h_2.b = b

#     permutation_samples = zeros(n_permutation) # zeros can be improved
#     pooled_atoms = vcat(h_1.atoms, h_2.atoms) # collect all rows
#         for i in 1:n_permutation
#             random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

#             atoms_1 = pooled_atoms[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
#             atoms_2 = pooled_atoms[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
#             h_1_permutation = HierSample(atoms_1, n, m, a, b)
#             h_2_permutation = HierSample(atoms_2, n, m, a, b)

#             permutation_samples[i] = ww(h_1_permutation, h_2_permutation)
#         end
#     threshold = quantile(permutation_samples, 1 - θ)

#     rej_rate = 0.0

#     @floop ThreadedEx() for s in 1:S
#         local h_1 = generate_hiersample(q_1, n, m)
#         local h_2 = generate_hiersample(q_2, n, m)
#         local a = minimum([h_1.a, h_2.a])
#         local b = maximum([h_1.b, h_2.b])
#         h_1.a = a
#         h_2.a = a
#         h_1.b = b
#         h_2.b = b
#         observed_test_stat = ww(h_1, h_2)
        
#         @reduce rej_rate += Float64(observed_test_stat > threshold)

#     end
#     return rej_rate / S
# end



# function rejection_rate_wow_permutation_parallel(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)
#     rej_rate = 0.0

#     @floop ThreadedEx() for s in 1:S
#         h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
#         a = minimum([h_1.a, h_2.a])
#         b = maximum([h_1.b, h_2.b])
#         h_1.a = a
#         h_2.a = a
#         h_1.b = b
#         h_2.b = b
#         observed_test_stat = ww(h_1, h_2)
        
#         # obtain quantile using permutation approach
#         permutation_samples = zeros(n_permutation) # zeros can be improved
        
#         pooled_atoms = vcat(h_1.atoms, h_2.atoms) # collect all rows
#         for i in 1:n_permutation
#             random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

#             atoms_1 = pooled_atoms[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
#             atoms_2 = pooled_atoms[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
#             h_1_permutation = HierSample(atoms_1, n, m, a, b)
#             h_2_permutation = HierSample(atoms_2, n, m, a, b)

#             permutation_samples[i] = ww(h_1_permutation, h_2_permutation)
#         end
#         threshold = quantile(permutation_samples, 1 - θ)
        
#         @reduce rej_rate += Float64(observed_test_stat > threshold)

#     end
#     return rej_rate / S
# end

