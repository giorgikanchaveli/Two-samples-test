using RCall # to call R functions
using FLoops # for parallel computing

include("distributions.jl")

include("distances/new_distance.jl")
include("distances/distance_Wasserstein.jl")

function test_statistic_energy(atoms_1::Matrix{Float64}, atoms_2::Matrix{Float64})
    # we assume that rows in each atoms are sorted.
    n = size(atoms_1)[1]
    
    distances_x = Matrix{Float64}(undef, n, n)
    distances_xy = Matrix{Float64}(undef, n, n)
    distances_y = Matrix{Float64}(undef, n, n)

    for i in 1:n
        x = atoms_1[i,:]
        y = atoms_2[i,:]
        for j in 1:n
            distances_x[i, j] = wasserstein1DUniform_sorted(x, atoms_1[j,:])
            distances_xy[i, j] = wasserstein1DUniform_sorted(x, atoms_2[j,:])
            distances_y[i, j] = wasserstein1DUniform_sorted(y, atoms_2[j,:])
        end
    end
    distance = 2 * mean(distances_xy) - mean(distances_x) - mean(distances_y)
    return distance * n / 2
end

function decide_energy(hier_sample_1::emp_ppm, hier_sample_2::emp_ppm, θ::Float64, n_samples::Int)
    n = hier_sample_1.n

    atoms_1 = hier_sample_1.atoms
    atoms_2 = hier_sample_2.atoms

    
    observed_test_stat = test_statistic_energy(atoms_1, atoms_2)
        
    # obtain quantile using bootstrap approach
    bootstrap_samples = zeros(n_samples) # zeros can be improved
    
    total_rows = vcat(atoms_1, atoms_2) # collect all rows
    for i in 1:n_samples
        indices_1 = sample(1:2*n, n; replace = true)
        indices_2 = sample(1:2*n, n; replace = true)
    
        bootstrap_samples[i] = test_statistic_energy(total_rows[indices_1,:], total_rows[indices_2,:])
    end
    threshold = quantile(bootstrap_samples, 1 - θ)
    
    return 1.0*(observed_test_stat > threshold)
end


function decide_dm(mu_1::Vector{Float64}, mu_2::Vector{Float64}, θ::Float64, n_bootstrap::Int)
    n = length(mu_1)
    
    @rput mu_1 mu_2 n n_bootstrap
    R"""

    library(frechet)
    n1 <- n
    n2 <- n
    delta <- 1
    qSup <- seq(0.01, 0.99, (0.99 - 0.01) / 50)

    Y1 <- lapply(1:n1, function(i) {
    qnorm(qSup, mu_1[i], sd = 1)
    })
    Y2 <- lapply(1:n2, function(i) {
    qnorm(qSup, mu_2[i], sd = 1)
    })
    Ly <- c(Y1, Y2)
    Lx <- qSup
    group <- c(rep(1, n1), rep(2, n2))
    res <- DenANOVA(qin = Ly, supin = Lx, group = group, optns = list(boot = TRUE, R = n_bootstrap))
    pvalue = res$pvalBoot # returns bootstrap pvalue
    """
    @rget pvalue  
    return 1 * (pvalue < θ)
end



function rejection_rate_dm(q_1::Union{tnormal_normal,simple_discr_1, simple_discr_2,mixture_ppm},
                             q_2::Union{tnormal_normal,simple_discr_1, simple_discr_2,mixture_ppm}, n::Int,
                         S::Int, θ::Float64, n_bootstrap::Int)
    rate = 0.0
    
    for i in 1:S
        # generate normal distributions 
        mu_1 = generate_prob_measures(q_1, n) # only contains means for normal distribution
        mu_2 = generate_prob_measures(q_2, n) # only contains means for normal distribution
        
        rate += decide_dm(mu_1, mu_2, θ, n_bootstrap) 
    end
    return rate/S
end




function threshold_hipm(hier_sample_1::emp_ppm, hier_sample_2::emp_ppm, θ::Float64, n_samples::Int, bootstrap::Bool)
    n = hier_sample_1.n
    m = hier_sample_1.m
    # set endpoints
    a = minimum((hier_sample_1.a, hier_sample_2.a))
    b = maximum((hier_sample_1.b, hier_sample_2.b))
    hier_sample_1.a = a
    hier_sample_2.a = a
    hier_sample_1.b = b
    hier_sample_2.b = b

    samples = zeros(n_samples)
    total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
    if bootstrap
        @floop ThreadedEx() for i in 1:n_samples
            indices_1 = sample(1:2*n, n; replace = true)
            indices_2 = sample(1:2*n, n; replace = true)

            atoms_1 = total_rows[indices_1,:] # first rows indexed by n random indices to the atoms_1
            atoms_2 = total_rows[indices_2,:] # first rows indexed by n random indices to the atoms_2

            hier_sample_1_bootstrap = emp_ppm(atoms_1, n, m, a, b)
            hier_sample_2_bootstrap = emp_ppm(atoms_2, n, m, a, b)

            samples[i] = dlip(hier_sample_1_bootstrap, hier_sample_2_bootstrap, a, b)
        end
    else
        @floop ThreadedEx() for i in 1:n_samples
            random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

            atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
            atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
            hier_sample_1_permutation = emp_ppm(atoms_1, n, m, a, b)
            hier_sample_2_permutation = emp_ppm(atoms_2, n, m, a, b)

            samples[i] = dlip(hier_sample_1_permutation, hier_sample_2_permutation, a,b)
        end
    end
    return quantile(samples, 1 - θ)
end



function threshold_wow(hier_sample_1::emp_ppm, hier_sample_2::emp_ppm, θ::Float64, n_samples::Int, bootstrap::Bool)
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






function rejection_rate_all(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_samples::Int, bootstrap::Bool)
    # if bootstrap is true then do bootstrap approach, n_samples refers to either number of permutations or bootstraps

    # firstly we obtain fixed thresholds for HIPM and WoW
    aux_hier_sample_1 = generate_emp(q_1,n,m)
    aux_hier_sample_2 = generate_emp(q_2, n, m)
    threshold_hipm_wrong = threshold_hipm(aux_hier_sample_1, aux_hier_sample_2, θ, n_samples, bootstrap) # gasaketebeli
    threshold_wow_wrong = threshold_wow(aux_hier_sample_1, aux_hier_sample_2, θ, n_samples, bootstrap) # gasaketebeli

    rates_hipm = 0.0
    rates_wow = 0.0
    rates_energy = 0.0

    @floop ThreadedEx() for s in 1:S
        # generate samples and set endpoints
        hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        a = minimum((hier_sample_1.a, hier_sample_2.a))
        b = maximum((hier_sample_1.b, hier_sample_2.b))
        hier_sample_1.a = a
        hier_sample_2.a = a
        hier_sample_1.b = b
        hier_sample_2.b = b

        # record decisions from each testing methods
        @reduce rates_hipm += 1.0*(dlip(hier_sample_1, hier_sample_2) > threshold_hipm_wrong)
        @reduce rates_wow += 1.0 * (ww(hier_sample_1, hier_sample_2) > threshold_wow_wrong)
        @reduce rates_energy += decide_energy(hier_sample_1, hier_sample_2, θ, n_samples) 
    end
    rates_energy /= S
    rates_wow /= S
    rates_hipm /= S
    rates_dm = 0.0
    rates_dm = rejection_rate_dm(q_1, q_2, n, S, θ, n_samples)
    return rates_hipm,rates_wow,rates_dm,rates_energy
end




function rejection_rate_wow(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int,
                     threshold_wow_wrong::Float64)
    # if bootstrap is true then do bootstrap approach, n_samples refers to either number of permutations or bootstraps



    rates_wow = 0.0
  
    @floop ThreadedEx() for s in 1:S
        # generate samples and set endpoints
        hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        

        # record decisions from each testing methods
        @reduce rates_wow += 1.0 * (ww(hier_sample_1, hier_sample_2) > threshold_wow_wrong)
    end
    rates_wow /= S
    return rates_wow
end

function rejection_rate_wow(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int,
                     θ::Float64, n_samples::Int, bootstrap::Bool)
    # firstly obtain threshold
    aux_hier_sample_1 = generate_emp(q_1,n,m)
    aux_hier_sample_2 = generate_emp(q_2, n, m)
    threshold_wow_wrong = threshold_wow(aux_hier_sample_1, aux_hier_sample_2, θ, n_samples, bootstrap) # gasaketebeli
    return rejection_rate_wow(q_1, q_2, n, m, S, threshold_wow_wrong)
end










function rejection_rate_hipm_wow(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_samples::Int, bootstrap::Bool)
    # if bootstrap is true then do bootstrap approach, n_samples refers to either number of permutations or bootstraps

    # firstly we obtain fixed thresholds for HIPM and WoW
    aux_hier_sample_1 = generate_emp(q_1,n,m)
    aux_hier_sample_2 = generate_emp(q_2, n, m)
    threshold_hipm_wrong = threshold_hipm(aux_hier_sample_1, aux_hier_sample_2, θ, n_samples, bootstrap) # gasaketebeli
    threshold_wow_wrong = threshold_wow(aux_hier_sample_1, aux_hier_sample_2, θ, n_samples, bootstrap) # gasaketebeli

    rates_hipm = 0.0
    rates_wow = 0.0

    @floop ThreadedEx() for s in 1:S
        # generate samples and set endpoints
        hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        a = minimum((hier_sample_1.a, hier_sample_2.a))
        b = maximum((hier_sample_1.b, hier_sample_2.b))
        hier_sample_1.a = a
        hier_sample_2.a = a
        hier_sample_1.b = b
        hier_sample_2.b = b

        # record decisions from each testing methods
        @reduce rates_hipm += 1.0*(dlip(hier_sample_1, hier_sample_2) > threshold_hipm_wrong)
        @reduce rates_wow += 1.0 * (ww(hier_sample_1, hier_sample_2) > threshold_wow_wrong)
    end
    rates_wow /= S
    rates_hipm /= S
    return rates_hipm,rates_wow
end



function save_fig_hipm_wow(pairs::Vector{<:Tuple{PPM,PPM}}, param_pairs::Vector{Float64}, file_name::String, file_path::String, title::String, xlabel::String, ylabel::String,
    n::Int, m::Int, S::Int, θ::Float64, n_samples::Int, bootstrap::Bool)
    rates_hipm = zeros(length(param_pairs))
    rates_wow = zeros(length(param_pairs))
    for i in 1:length(pairs)
        q_1, q_2 = pairs[i]
        r_hipm, r_wow = rejection_rate_hipm_wow(q_1,q_2,n,m,S,θ,n_samples,bootstrap)
        rates_hipm[i] = r_hipm
        rates_wow[i] = r_wow
        println(i)
    end
    fig = plot(title = title, xlabel = xlabel, ylabel = ylabel, xlims=(minimum(param_pairs) - 0.10, maximum(param_pairs)+ 0.10),
                         ylims = (-0.1, 1.1))
    plot!(fig, param_pairs, rates_hipm, label = "hipm", color = "green", marker = (:circle, 4))
    plot!(fig, param_pairs, rates_wow, label = "wow", color = "brown", marker = (:circle, 4))
    filepath = joinpath(pwd(), file_path)
    savefig(fig,joinpath(filepath, file_name))
end


# up to now more or less everything is fine



# function rejection_rate_energy_boostrap_parallel(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)
#     rej_rate = 0.0

#     @floop ThreadedEx() for s in 1:S
#         hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
#         observed_test_stat = test_statistic_energy(hier_sample_1, hier_sample_2)
        
#         # obtain quantile using bootstrap approach
#         boostrap_samples = zeros(n_boostrap) # zeros can be improved
#         a = minimum([hier_sample_1.a, hier_sample_2.a])
#         b = maximum([hier_sample_1.b, hier_sample_2.b])
#         total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
#         for i in 1:n_boostrap
#             indices_1 = sample(1:2*n, n; replace = true)
#             indices_2 = sample(1:2*n, n; replace = true)
#             atoms_1 = total_rows[indices_1,:]  # resample from pooled hierarchical sample
#             atoms_2 = total_rows[indices_2,:]  # resample from pooled hierarchical sample
            
        
#             hier_sample_1_boostrap = emp_ppm(atoms_1, n, m, a, b)
#             hier_sample_2_boostrap = emp_ppm(atoms_2, n, m, a, b)

#             boostrap_samples[i] = test_statistic_energy(hier_sample_1_boostrap, hier_sample_2_boostrap)
#         end
#         threshold = quantile(boostrap_samples, 1 - θ)
        
#         @reduce rej_rate += 1.0*(observed_test_stat > threshold)
#     end
#     return rej_rate / S
# end



# function decide_dm(hier_sample_1::emp_ppm, hier_sample_2::emp_ppm, θ::Float64, n_samples::Int)
#     atoms_1 = copy(hier_sample_1.atoms)
#     atoms_2 = copy(hier_sample_2.atoms)
#     n = hier_sample_1.n     
    
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
#     return 1 * (pvalue < θ)
# end



# function rejection_rate_dm_boostrap(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)

#     rej_rate = 0.0
#     for s in 1:S
     
#         hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
#         atoms_1, atoms_2 = hier_sample_1.atoms, hier_sample_2.atoms

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
#         rej_rate += 1 * (pvalue < θ)
#     end
#     rej_rate /= S
#     return rej_rate
# end



# function rejection_rate_hipm_boostrap_parallel(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)
#     rej_rate = 0.0

#     @floop ThreadedEx() for s in 1:S
#         hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
#         a = minimum([hier_sample_1.a, hier_sample_2.a])
#         b = maximum([hier_sample_1.b, hier_sample_2.b])
#         hier_sample_1.a = a
#         hier_sample_2.a = a
#         hier_sample_1.b = b
#         hier_sample_2.b = b
#         observed_test_stat = dlip(hier_sample_1, hier_sample_2)
        
#         # obtain quantile using bootstrap approach
#         boostrap_samples = zeros(n_boostrap) # zeros can be improved
        
#         total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
#         for i in 1:n_boostrap
#             indices_1 = sample(1:2*n, n; replace = true)
#             indices_2 = sample(1:2*n, n; replace = true)
#             atoms_1 = total_rows[indices_1,:]  # resample from pooled hierarchical sample
#             atoms_2 = total_rows[indices_2,:]  # resample from pooled hierarchical sample
            
        
#             hier_sample_1_boostrap = emp_ppm(atoms_1, n, m, a, b)
#             hier_sample_2_boostrap = emp_ppm(atoms_2, n, m, a, b)

#             boostrap_samples[i] = dlip(hier_sample_1_boostrap, hier_sample_2_boostrap)
#         end
#         threshold = quantile(boostrap_samples, 1 - θ)
        
#         @reduce rej_rate += 1.0*(observed_test_stat > threshold)
#     end
#     return rej_rate / S
# end


# function rejection_rate_hipm_permutation_wrong(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)

#     # firstly we obtain threshold
#     hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
#     a = minimum([hier_sample_1.a, hier_sample_2.a])
#     b = maximum([hier_sample_1.b, hier_sample_2.b])
#     hier_sample_1.a = a
#     hier_sample_2.a = a
#     hier_sample_1.b = b
#     hier_sample_2.b = b

#     permutation_samples = zeros(n_permutation) # zeros can be improved
#     total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
#         for i in 1:n_permutation
#             random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

#             atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
#             atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
#             hier_sample_1_permutation = emp_ppm(atoms_1, n, m, a, b)
#             hier_sample_2_permutation = emp_ppm(atoms_2, n, m, a, b)

#             permutation_samples[i] = dlip(hier_sample_1_permutation, hier_sample_2_permutation)
#         end
#     threshold = quantile(permutation_samples, 1 - θ)

#     rej_rate = 0.0

#     @floop ThreadedEx() for s in 1:S
#         local hier_sample_1 = generate_emp(q_1, n, m)
#         local hier_sample_2 = generate_emp(q_2, n, m)
#         local a = minimum([hier_sample_1.a, hier_sample_2.a])
#         local b = maximum([hier_sample_1.b, hier_sample_2.b])
#         hier_sample_1.a = a
#         hier_sample_2.a = a
#         hier_sample_1.b = b
#         hier_sample_2.b = b
#         observed_test_stat = dlip(hier_sample_1, hier_sample_2)
        
#         @reduce rej_rate += 1.0*(observed_test_stat > threshold)
#     end
#     return rej_rate / S
# end



# function rejection_rate_hipm_permutation_parallel(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)
#     rej_rate = 0.0

#     @floop ThreadedEx() for s in 1:S
#         hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
#         a = minimum([hier_sample_1.a, hier_sample_2.a])
#         b = maximum([hier_sample_1.b, hier_sample_2.b])
#         hier_sample_1.a = a
#         hier_sample_2.a = a
#         hier_sample_1.b = b
#         hier_sample_2.b = b
#         observed_test_stat = dlip(hier_sample_1, hier_sample_2)
        
#         # obtain quantile using permutation approach
#         permutation_samples = zeros(n_permutation) # zeros can be improved
        
#         total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
#         for i in 1:n_permutation
#             random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

#             atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
#             atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
#             hier_sample_1_permutation = emp_ppm(atoms_1, n, m, a, b)
#             hier_sample_2_permutation = emp_ppm(atoms_2, n, m, a, b)

#             permutation_samples[i] = dlip(hier_sample_1_permutation, hier_sample_2_permutation)
#         end
#         threshold = quantile(permutation_samples, 1 - θ)
        
#         @reduce rej_rate += 1.0*(observed_test_stat > threshold)
#     end
#     return rej_rate / S
# end



# function rejection_rate_wow_boostrap_parallel(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)
#     rej_rate = 0.0

#     @floop ThreadedEx() for s in 1:S
#         hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
#         a = minimum([hier_sample_1.a, hier_sample_2.a])
#         b = maximum([hier_sample_1.b, hier_sample_2.b])
#         hier_sample_1.a = a
#         hier_sample_2.a = a
#         hier_sample_1.b = b
#         hier_sample_2.b = b
#         observed_test_stat = ww(hier_sample_1, hier_sample_2)
        
#         # obtain quantile using bootstrap approach
#         boostrap_samples = zeros(n_boostrap) # zeros can be improved
  
#         total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
#         for i in 1:n_boostrap
#             indices_1 = sample(1:2*n, n; replace = true)
#             indices_2 = sample(1:2*n, n; replace = true)
#             atoms_1 = total_rows[indices_1,:]  # resample from pooled hierarchical sample
#             atoms_2 = total_rows[indices_2,:]  # resample from pooled hierarchical sample
            
        
#             hier_sample_1_boostrap = emp_ppm(atoms_1, n, m, a, b)
#             hier_sample_2_boostrap = emp_ppm(atoms_2, n, m, a, b)

#             boostrap_samples[i] = ww(hier_sample_1_boostrap, hier_sample_2_boostrap)
#         end
#         threshold = quantile(boostrap_samples, 1 - θ)
        
#         @reduce rej_rate += 1.0*(observed_test_stat > threshold)
#     end
#     return rej_rate / S
# end


# function rejection_rate_wow_permutation_wrong(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)

#     # firstly we obtain threshold
#     hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
#     a = minimum([hier_sample_1.a, hier_sample_2.a])
#     b = maximum([hier_sample_1.b, hier_sample_2.b])
#     hier_sample_1.a = a
#     hier_sample_2.a = a
#     hier_sample_1.b = b
#     hier_sample_2.b = b

#     permutation_samples = zeros(n_permutation) # zeros can be improved
#     total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
#         for i in 1:n_permutation
#             random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

#             atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
#             atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
#             hier_sample_1_permutation = emp_ppm(atoms_1, n, m, a, b)
#             hier_sample_2_permutation = emp_ppm(atoms_2, n, m, a, b)

#             permutation_samples[i] = ww(hier_sample_1_permutation, hier_sample_2_permutation)
#         end
#     threshold = quantile(permutation_samples, 1 - θ)

#     rej_rate = 0.0

#     @floop ThreadedEx() for s in 1:S
#         local hier_sample_1 = generate_emp(q_1, n, m)
#         local hier_sample_2 = generate_emp(q_2, n, m)
#         local a = minimum([hier_sample_1.a, hier_sample_2.a])
#         local b = maximum([hier_sample_1.b, hier_sample_2.b])
#         hier_sample_1.a = a
#         hier_sample_2.a = a
#         hier_sample_1.b = b
#         hier_sample_2.b = b
#         observed_test_stat = ww(hier_sample_1, hier_sample_2)
        
#         @reduce rej_rate += 1.0*(observed_test_stat > threshold)
#     end
#     return rej_rate / S
# end



# function rejection_rate_wow_permutation_parallel(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)
#     rej_rate = 0.0

#     @floop ThreadedEx() for s in 1:S
#         hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
#         a = minimum([hier_sample_1.a, hier_sample_2.a])
#         b = maximum([hier_sample_1.b, hier_sample_2.b])
#         hier_sample_1.a = a
#         hier_sample_2.a = a
#         hier_sample_1.b = b
#         hier_sample_2.b = b
#         observed_test_stat = ww(hier_sample_1, hier_sample_2)
        
#         # obtain quantile using permutation approach
#         permutation_samples = zeros(n_permutation) # zeros can be improved
        
#         total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
#         for i in 1:n_permutation
#             random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

#             atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
#             atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
#             hier_sample_1_permutation = emp_ppm(atoms_1, n, m, a, b)
#             hier_sample_2_permutation = emp_ppm(atoms_2, n, m, a, b)

#             permutation_samples[i] = ww(hier_sample_1_permutation, hier_sample_2_permutation)
#         end
#         threshold = quantile(permutation_samples, 1 - θ)
        
#         @reduce rej_rate += 1.0*(observed_test_stat > threshold)
#     end
#     return rej_rate / S
# end

