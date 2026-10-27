# This file contains functions for simulations using DM (Fréchet), Energy, HIPM and WoW tests, and for mortality dataset.
# In particular, functions for 
    # 1) test statistics
    # 2) decision of rejection
    # 3) thresholds
    # 4) rejection rates.
    # 5) p-value calculations.


# using RCall # to call R functions (only needed for the commented-out decision_dm and rejection_rate_dm)
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

Given two hierarchical samples, decides whether to reject null hypothesis using the method from Szekely et al. (2004)
via bootstrap/permutation approach. For permutation, the p-value counts the identity permutation, i.e. it is
(1 + #{T_b >= T}) / (n_samples + 1).

# Arguments:
    h_1::HierSample
    h_2::HierSample
    θ::Float64       :  Significance level
    n_samples::Int   :  number of bootstrap/permutation samples
    bootstrap::Bool  :  Boolean variable. If true use bootstrap, otherwise permutation.
"""

function decision_energy(h_1::HierSample, h_2::HierSample, θ::Float64, n_samples::Int, bootstrap::Bool)

    atoms_1 = h_1.atoms
    atoms_2 = h_2.atoms
    n = size(atoms_1)[1]

    observed_test_stat = test_statistic_energy(atoms_1, atoms_2)

    # obtain bootstrap/permutation samples
    samples = zeros(n_samples)
    pooled_atoms = vcat(atoms_1, atoms_2) # collect all rows
    if bootstrap
        for i in 1:n_samples
            indices_1 = sample(1:2*n, n; replace = true)
            indices_2 = sample(1:2*n, n; replace = true)

            new_atoms_1 = @view pooled_atoms[indices_1,:]
            new_atoms_2 = @view pooled_atoms[indices_2,:]
            samples[i] = test_statistic_energy(new_atoms_1, new_atoms_2)
        end
        pvalue = mean(samples .>= observed_test_stat)
    else
        for i in 1:n_samples
            random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

            new_atoms_1 = @view pooled_atoms[random_indices[1:n],:] # rows indexed by first n random indices to the atoms_1
            new_atoms_2 = @view pooled_atoms[random_indices[n+1:end],:] # rows indexed by the rest of random indices to the atoms_2
            samples[i] = test_statistic_energy(new_atoms_1, new_atoms_2)
        end
        pvalue = (1 + count(t -> t >= observed_test_stat, samples)) / (n_samples + 1) # counting the identity permutation
    end
    return Float64(pvalue < θ)
end


# DM test computed with frechet::DenANOVA in R (via RCall). Not used anymore: rejection_rate_all uses decision_frechet
# below, which runs on the same hierarchical samples as the other tests. To use these functions again, uncomment
# this block and `using RCall` at the top of the file.
#=
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

    length(means_1) == length(means_2) || throw(ArgumentError("means_1 and means_2 must have the same length. Got $(length(means_1)) and $(length(means_2))."))
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

Given laws of RPMS , estimates the rejection rate of the testing scheme
according to Dubey and Muller (2019) using hierarchical samples.  

# Arguments:
    q_1::LawRPM                                            : law of RPM
    q_2::LawRPM                                            : law of RPM
    n::Int                                                 :  number of probability measures
    m::Int                                                 :  length of exchangeable sequence
    S::Int                                                 :  number of MCMC iterations to estimate rejection rate
    θ::Float64                                             :  significance level
    n_bootstrap::Int                                       :  number of bootstrap samples.
"""

function rejection_rate_dm(q_1::LawRPM,
                             q_2::LawRPM, 
                             n::Int, m::Int, S::Int, θ::Float64, n_bootstrap::Int)
    rej_rate = 0.0
    
    for _ in 1:S
        # generate hierarchical samples
        h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
        
        rej_rate += decision_dm(h_1, h_2, θ, n_bootstrap) 
    end
    return rej_rate/S
end


"""
    rejection_rate_dm

Given specific laws of RPMS (laws on N(δ, 1)), estimates the rejection rate of the testing scheme
according to Dubey and Muller (2019). It works directly on probability measures instead of hierarchical samples.

# Arguments:
    q_1::Union{tnormal_normal, discr_normal, mixture}  :  law of RPM
    q_2::Union{tnormal_normal, discr_normal, mixture}  :  law of RPM
    n::Int                                                 :  number of probability measures
    m::Int                                                 :  length of exchangeable sequence
    S::Int                                                 :  number of MCMC iterations to estimate rejection rate
    θ::Float64                                             :  significance level
    n_bootstrap::Int                                       :  number of bootstrap samples.
"""

function rejection_rate_dm(q_1::Union{tnormal_normal, discr_normal, mixture},
                             q_2::Union{tnormal_normal, discr_normal, mixture}, 
                             n::Int, m::Int, S::Int, θ::Float64, n_bootstrap::Int)
    rej_rate = 0.0
    
    for i in 1:S
        # generate normal distributions 
        means_1 = generate_prob_measures(q_1, n) # only contains means for normal distribution
        means_2 = generate_prob_measures(q_2, n) # only contains means for normal distribution
        
        rej_rate += decision_dm(means_1, means_2, θ, n_bootstrap)
    end
    return rej_rate/S
end
=#


# Fréchet test from Dubey and Muller (2019), implemented in Julia and applied directly to two hierarchical samples.


# Important Notes:
#        1) In 1D, the 2-Wasserstein distance between two probability measures is the L^2 distance between their
#  quantile functions. Since every row has the same number of atoms m, the quantile function of a row is given
#  by its sorted atoms. Hence the squared distance between rows x and y is (1/m) * Σ_k (x_(k) - y_(k))^2, and the
#  Fréchet mean (Wasserstein barycentre) of several rows is the sequence whose sorted atoms are the
#  column-wise average of the rows.
#        2) Why we estimate quantile functions directly instead of estimating densities first (as frechet::DenANOVA
#  in R does with yin, see the commented-out decision_dm above):
#           - The test statistic depends on each probability measure only through its quantile function, so a
#             density is an unnecessary intermediate step. The sorted atoms of a row already consistently estimate
#             the quantile function of its latent probability measure, and with them every distance and Fréchet mean
#             is computed exactly (no integration grid, no interpolation).
#           - Density estimation adds choices that change the result: kernel, bandwidth (DenANOVA uses twice the
#             default bandwidth, i.e. oversmooths), and support (DenANOVA truncates it to [min, max] of each row).
#             Smoothing inflates the variance of each measure, so the test no longer compares the measures that
#             generated the data.
#           - Some laws of RPM in distributions.jl (DP, unif_spike, discr_law with discrete components) generate
#             measures without a density, so rows contain repeated atoms. There, density estimation is ill-defined
#             and DenANOVA may fail, while quantile functions are always well-defined.
#           - This way, the Fréchet test uses the rows exactly as HIPM, WoW and Energy tests do.
#        3) Rows of the hierarchical samples must be sorted, as for WoW. Hierarchical samples generated by
#  generate_hiersample in distributions.jl already satisfy this. Sortedness is not checked.
#        4) Calibration of p-values:
#           - Permutation: the p-value (1 + #{T_b >= T}) / (n_samples + 1) is valid in finite samples whenever the
#             rows of both hierarchical samples are exchangeable under the null. The null of Dubey and Muller (2019),
#             equality of Fréchet means and Fréchet variances, does not imply this. For hierarchical samples,
#             exchangeability holds when both laws of RPM are equal and the rows are generated by the same inner
#             sampling scheme (same m, atoms i.i.d. given the latent measure), as in generate_hiersample.
#           - Bootstrap: the p-value #{T_b >= T} / n_samples is only asymptotically valid.
#        5) Degenerate statistic: T_n is undefined when σ2_1 = σ2_2 = 0, i.e. when in both groups all rows are at the
#  same squared distance from their group's Fréchet mean (e.g. a group with exactly two rows, or with identical rows).
#  Then we throw a DomainError instead of adding an epsilon or discarding the sample. This can happen in a
#  bootstrap/permutation replicate even when the observed samples are not degenerate; the error names the replicate.
#  Because of rounding, σ2 of a group with identical rows can be tiny but nonzero (e.g. 1e-98) instead of exactly 0.
#  Such samples are not detected and give a very large T_n.


"""
    check_atoms_frechet

Checks that two matrices of atoms are valid inputs for the Fréchet test: at least two rows each, the same positive
number of columns, and finite atoms.

# Arguments:
    atoms_1::AbstractArray{Float64, 2}
    atoms_2::AbstractArray{Float64, 2}
"""
function check_atoms_frechet(atoms_1::AbstractArray{Float64, 2}, atoms_2::AbstractArray{Float64, 2})
    n_1, m_1 = size(atoms_1)
    n_2, m_2 = size(atoms_2)
    (n_1 >= 2 && n_2 >= 2) || throw(ArgumentError("Each hierarchical sample must have at least two rows. Got $n_1 and $n_2."))
    m_1 >= 1 || throw(ArgumentError("Hierarchical samples must have a positive number of columns. Got $m_1."))
    m_1 == m_2 || throw(ArgumentError("Both hierarchical samples must have the same number of columns. Got $m_1 and $m_2."))
    (all(isfinite, atoms_1) && all(isfinite, atoms_2)) || throw(ArgumentError("All atoms must be finite."))
    return nothing
end


"""
    squared_distances_to_mean

Given a matrix whose rows are sorted exchangeable sequences, returns the squared 2-Wasserstein distance of
each row to the Fréchet mean of all rows.

# Arguments:
    atoms::AbstractArray{Float64, 2}  :  each row is a sorted exchangeable sequence.

# Warning: This function assumes that atoms are sorted.
"""
function squared_distances_to_mean(atoms::AbstractArray{Float64, 2})
    frechet_mean = mean(atoms, dims = 1) # sorted atoms of the Fréchet mean
    return vec(mean((atoms .- frechet_mean).^2, dims = 2))
end


"""
    frechet_statistic

Given two matrices of atoms (hierarchical sample using definition in paper), computes the test statistic T_n from
Dubey and Muller (2019) for two groups. Inputs are not validated (pvalue_frechet does this with check_atoms_frechet).

# Arguments:
    atoms_1::AbstractArray{Float64, 2}
    atoms_2::AbstractArray{Float64, 2}
    context::String                     :  describes the samples (e.g. "bootstrap replicate 7") in error messages.

# Warning:
    Each row in matrices atoms_1/2 must be sorted.
"""
function frechet_statistic(atoms_1::AbstractArray{Float64, 2}, atoms_2::AbstractArray{Float64, 2}, context::String)

    n_1 = size(atoms_1)[1]
    n_2 = size(atoms_2)[1]
    n_total = n_1 + n_2
    λ_1, λ_2 = n_1 / n_total, n_2 / n_total

    distances_1 = squared_distances_to_mean(atoms_1)
    distances_2 = squared_distances_to_mean(atoms_2)

    # Fréchet variances
    V_1 = mean(distances_1)
    V_2 = mean(distances_2)

    # variances of squared distances to the Fréchet mean (denominator n_j, as in Dubey and Muller (2019))
    σ2_1 = mean((distances_1 .- V_1).^2)
    σ2_2 = mean((distances_2 .- V_2).^2)

    if σ2_1 == 0 && σ2_2 == 0
        throw(DomainError((σ2_1, σ2_2), "Fréchet test statistic is undefined for $context: σ2_1 = σ2_2 = 0, i.e. in both " *
            "groups all rows are at the same squared Wasserstein distance from their group's Fréchet mean " *
            "(e.g. groups with exactly two rows or with identical rows)."))
    end

    # between-group variation, equal to V_pooled - λ_1 * V_1 - λ_2 * V_2 but without cancellation
    mean_1 = mean(atoms_1, dims = 1)
    mean_2 = mean(atoms_2, dims = 1)
    F_n = λ_1 * λ_2 * mean((mean_1 .- mean_2).^2)

    variance_term = (V_1 - V_2)^2 / (σ2_1 / n_1 + σ2_2 / n_2) # large when Fréchet variances differ
    mean_term = n_total * F_n^2 / (λ_1^2 * σ2_1 + λ_2^2 * σ2_2) # large when Fréchet means differ
    statistic = variance_term + mean_term

    isfinite(statistic) || throw(DomainError(statistic, "Fréchet test statistic is not finite for $context."))
    return statistic
end


"""
    pvalue_frechet

Given two hierarchical samples, computes p-value for the Fréchet test from Dubey and Muller (2019).
For permutation, the p-value is (1 + #{T_b >= T}) / (n_samples + 1), valid in finite samples when the rows are
exchangeable under the null. For bootstrap, it is #{T_b >= T} / n_samples, which is only asymptotically valid.

# Arguments:
    h_1::HierSample
    h_2::HierSample
    n_samples::Int   :  number of bootstrap/permutation samples
    bootstrap::Bool  :  Boolean variable. If true use bootstrap, otherwise permutation.

# Warning:
    This function assumes that atoms are sorted.
    Throws a DomainError if the observed samples or a bootstrap/permutation replicate are degenerate.
"""
function pvalue_frechet(h_1::HierSample, h_2::HierSample, n_samples::Int, bootstrap::Bool)

    atoms_1 = h_1.atoms
    atoms_2 = h_2.atoms
    check_atoms_frechet(atoms_1, atoms_2)
    n_samples >= 1 || throw(ArgumentError("n_samples must be positive. Got $n_samples."))

    n_1 = size(atoms_1)[1]
    n_2 = size(atoms_2)[1]
    n_total = n_1 + n_2

    observed_test_stat = frechet_statistic(atoms_1, atoms_2, "the observed hierarchical samples")

    # obtain bootstrap/permutation samples
    samples = zeros(n_samples)
    pooled_atoms = vcat(atoms_1, atoms_2) # collect all rows

    if bootstrap
        for i in 1:n_samples
            indices_1 = sample(1:n_total, n_1; replace = true)
            indices_2 = sample(1:n_total, n_2; replace = true)

            new_atoms_1 = @view pooled_atoms[indices_1,:]
            new_atoms_2 = @view pooled_atoms[indices_2,:]

            samples[i] = frechet_statistic(new_atoms_1, new_atoms_2, "bootstrap replicate $i")
        end
        return mean(samples .>= observed_test_stat) # pvalue
    else
        for i in 1:n_samples
            random_indices = randperm(n_total) # indices to distribute rows to new hierarchical meausures

            new_atoms_1 = @view pooled_atoms[random_indices[1:n_1],:] # rows indexed by first n_1 random indices to the atoms_1
            new_atoms_2 = @view pooled_atoms[random_indices[n_1+1:end],:] # rows indexed by the rest of random indices to the atoms_2

            samples[i] = frechet_statistic(new_atoms_1, new_atoms_2, "permutation replicate $i")
        end
        return (1 + count(t -> t >= observed_test_stat, samples)) / (n_samples + 1) # pvalue, counting the identity permutation
    end
end


"""
    decision_frechet

Given two hierarchical samples, decides whether to reject H_0 using the Fréchet test from Dubey and Muller (2019)
via bootstrap/permutation approach. Returns 1.0 if H_0 is rejected and 0.0 otherwise.

# Arguments:
    h_1::HierSample
    h_2::HierSample
    θ::Union{Float64, Vector{Float64}}  :  Significance level/s
    n_samples::Int                      :  number of bootstrap/permutation samples
    bootstrap::Bool                     :  Boolean variable. If true use bootstrap, otherwise permutation.

# Warning:
    This function assumes that atoms are sorted.
    Throws a DomainError if the observed samples or a bootstrap/permutation replicate are degenerate.
"""
function decision_frechet(h_1::HierSample, h_2::HierSample, θ::Union{Float64, Vector{Float64}}, n_samples::Int, bootstrap::Bool)
    pvalue = pvalue_frechet(h_1, h_2, n_samples, bootstrap)
    return Float64.(pvalue .< θ)
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
function threshold_hipm(h_1::HierSample, h_2::HierSample, θ::Union{Float64, Vector{Float64}}, n_samples::Int, bootstrap::Bool; n_rerun::Int = 10)
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

            samples[i] = dlip(new_atoms_1, new_atoms_2, a, b; n_rerun)
        end
    else
        for i in 1:n_samples
            random_indices = randperm(n_total) # indices to distribute rows to new hierarchical meausures

            new_atoms_1 = @view pooled_atoms[random_indices[1:n_1],:] # rows indexed by first n_1 random indices to the atoms_1
            new_atoms_2 = @view pooled_atoms[random_indices[n_1+1:end],:] # rows indexed by the rest of random indices to the atoms_2

            samples[i] = dlip(new_atoms_1, new_atoms_2, a, b; n_rerun)
        end
    end
    return quantile(samples, 1 .- θ) .* sqrt(n_1*n_2 / n_total)
end


"""
    pvalue_hipm

Given two hierarchical samples, computes p-value for HIPM. For permutation, the p-value counts the identity
permutation, i.e. it is (1 + #{T_b >= T}) / (n_samples + 1).

# Arguments:
    h_1::HierSample
    h_2::HierSample
    n_samples::Int   :  number of bootstrap/permutation samples
    bootstrap::Bool  :  Boolean variable. If true use bootstrap, otherwise permutation.
"""
function pvalue_hipm(h_1::HierSample, h_2::HierSample, n_samples::Int, bootstrap::Bool)
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
    if bootstrap
        return mean(samples .>= observed_distance) # pvalue
    else
        return (1 + count(t -> t >= observed_distance, samples)) / (n_samples + 1) # pvalue, counting the identity permutation
    end
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
    pvalue = pvalue_hipm(h_1, h_2, n_samples, bootstrap)
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
    pvalue_wow

Given two hierarchical samples, computes p-value for WoW. For permutation, the p-value counts the identity
permutation, i.e. it is (1 + #{T_b >= T}) / (n_samples + 1).

# Arguments:
    h_1::HierSample
    h_2::HierSample
    n_samples::Int   :  number of bootstrap/permutation samples
    bootstrap::Bool  :  Boolean variable. If true use bootstrap, otherwise permutation.
"""
function pvalue_wow(h_1::HierSample, h_2::HierSample, n_samples::Int, bootstrap::Bool)
    
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
    if bootstrap
        return mean(samples .>= observed_distance) # pvalue
    else
        return (1 + count(t -> t >= observed_distance, samples)) / (n_samples + 1) # pvalue, counting the identity permutation
    end
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
    
    pvalue = pvalue_wow(h_1, h_2, n_samples, bootstrap)
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

Given two laws of RPMs, returns rejection rates for all 4 testing schemes: HIPM, WoW, DM, Energy. All 4 tests record
decisions on the same hierarchical samples. DM is the Fréchet test from Dubey and Muller (2019) computed by decision_frechet.

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
    rates_dm = 0.0
    rates_energy = 0.0

    @floop ThreadedEx() for s in 1:S
        # generate samples and set endpoints
        h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)

        # record decisions from each testing methods
        @reduce rates_hipm += decision_hipm(h_1, h_2, θ, n_samples, bootstrap)
        @reduce rates_wow += decision_wow(h_1, h_2, θ, n_samples, bootstrap)
        @reduce rates_dm += decision_frechet(h_1, h_2, θ, n_samples, bootstrap)
        @reduce rates_energy += decision_energy(h_1, h_2, θ, n_samples, bootstrap)
    end
    rates_energy /= S
    rates_dm /= S
    rates_wow /= S
    rates_hipm /= S

    return rates_hipm,rates_wow,rates_dm,rates_energy
end


# Codes below are for the Mortality dataset.


"""

pvalue_hipm

This function is specifically for mortality dataset. Given atoms and weights associated to two Hierarchical estimators,
we estimate the p-value for HIPM via permutation approach. 

# Arguments:
    atoms_1::Matrix{Float64}
    weights_1::Matrix{Float64}
    atoms_2::Matrix{Float64}
    weights_2::Matrix{Float64}
    n_permutations::Int
    max_time::Float  :  number of seconds for runnnig optimizaiton algorithm in HIPM
"""

function pvalue_hipm(atoms_1::Matrix{Float64}, weights_1::Matrix{Float64}, atoms_2::Matrix{Float64}, weights_2::Matrix{Float64},
                     n_permutations::Int, max_time::Float64)
    n_1 = size(atoms_1,1)
    n_2 = size(atoms_2,1)
    n = n_1 + n_2
    a = atoms_1[1,1] # matrices for atoms have 0,1,...,age_truncation in each rows.
    b = atoms_1[1,end]
    
    T_observed = dlip(atoms_1, atoms_2, weights_1, weights_2, a, b; max_time = max_time)
  
    samples = zeros(n_permutations)
    total_weights = vcat(weights_1, weights_2) # collect all rows

    @floop ThreadedEx() for i in 1:n_permutations
        random_indices = randperm(n) # indices to distribute rows to new hierarchical meausures

        new_weights_1 = view(total_weights, random_indices[1:n_1], :) # first rows indexed by n random indices to the atoms_1
        new_weights_2 = view(total_weights, random_indices[n_1+1:end], :) # first rows indexed by n random indices to the atoms_2

        samples[i] = dlip(atoms_1, atoms_2, new_weights_1, new_weights_2, a, b; max_time = max_time)
    end 
    return (1 + count(t -> t >= T_observed, samples)) / (n_permutations + 1) # pvalue, counting the identity permutation
end 





"""

pvalue_wow

This function is specifically for mortality dataset. Given hierarchical sample obtained from death counts,
we estimate the p-value for WoW via permutation approach. 

# Arguments:
    atoms_1::Vector{Vector{Float64}}
    atoms_2::Vector{Vector{Float64}}
    n_permutations::Int
"""

function pvalue_wow(atoms_1::Vector{Vector{Float64}}, atoms_2::Vector{Vector{Float64}},
                     n_permutations::Int)
    n_1 = length(atoms_1)
    n_2 = length(atoms_2)
    n = n_1 + n_2
   
    T_observed = ww(atoms_1, atoms_2)
  
    samples = zeros(n_permutations)
    total_atoms = vcat(atoms_1, atoms_2) # collect all rows

    @floop ThreadedEx() for i in 1:n_permutations
        random_indices = randperm(n) # indices to distribute rows to new hierarchical meausures
        new_atoms_1 = [total_atoms[i] for i in random_indices[1:n_1]]
        new_atoms_2 = [total_atoms[i] for i in random_indices[n_1+1:end]]

        samples[i] = ww(new_atoms_1, new_atoms_2)
    end 
    return (1 + count(t -> t >= T_observed, samples)) / (n_permutations + 1) # pvalue, counting the identity permutation
end 





"""

pvalue_averaging

This function is specifically for mortality dataset. Given probability measures in each group, we average them inside groups and then use
Wasserstein-1 distance.

# Arguments:
    atoms_1::Matrix{Float64}
    weights_1::Matrix{Float64}
    atoms_2::Matrix{Float64}
    weights_2::Matrix{Float64}
    n_permutations::Int
"""

function pvalue_averaging(atoms_1::Matrix{Float64}, weights_1::Matrix{Float64}, atoms_2::Matrix{Float64}, weights_2::Matrix{Float64},
     n_permutations::Int)
    n_1 = size(atoms_1,1)
    n_2 = size(atoms_2,1)
    n = n_1 + n_2
    a = atoms_1[1,1]
    b = atoms_1[1,end]

    # compute observed distance
    # Note that atoms are same for each measure and spacing between them is 1.
    average_weights_1 = vec(mean(weights_1, dims = 1))
    average_weights_2 = vec(mean(weights_2, dims = 1))
    T_observed = sum(abs.(cumsum(average_weights_1) .- cumsum(average_weights_2)))

    # obtain samples of distances using Permutation approach
    samples = zeros(n_permutations)
    total_weights = vcat(weights_1, weights_2) # collect all rows

    for i in 1:n_permutations
        random_indices = randperm(n) # indices to distribute rows to new hierarchical meausures

        new_weights_1 = view(total_weights, random_indices[1:n_1], :) # first rows indexed by n random indices to the atoms_1
        new_weights_2 = view(total_weights, random_indices[n_1+1:end], :) # first rows indexed by n random indices to the atoms_2
        average_weights_1 = vec(mean(new_weights_1, dims = 1))
        average_weights_2 = vec(mean(new_weights_2, dims = 1))
        samples[i] = sum(abs.(cumsum(average_weights_1) .- cumsum(average_weights_2)))
    end
    return (1 + count(t -> t >= T_observed, samples)) / (n_permutations + 1) # pvalue, counting the identity permutation
end 







