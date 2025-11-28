using RCall # to call R functions
using Plots

include("distributions.jl")

include("distances/new_distance.jl")
include("distances/distance_Wasserstein.jl")
using DataFrames
using CSV
using FLoops

include("methods.jl")

# function test_statistic_energy(atoms_1::Matrix{Float64}, atoms_2::Matrix{Float64})
#     # we assume that rows in each atoms are sorted.
#     n = size(atoms_1)[1]
    
#     distances_x = Matrix{Float64}(undef, n, n)
#     distances_xy = Matrix{Float64}(undef, n, n)
#     distances_y = Matrix{Float64}(undef, n, n)

#     for i in 1:n
#         x = atoms_1[i,:]
#         y = atoms_2[i,:]
#         for j in 1:n
#             distances_x[i, j] = wasserstein1DUniform_sorted(x, atoms_1[j,:], 1)
#             distances_xy[i, j] = wasserstein1DUniform_sorted(x, atoms_2[j,:], 1)
#             distances_y[i, j] = wasserstein1DUniform_sorted(y, atoms_2[j,:], 1)
#         end
#     end
#     distance = 2 * mean(distances_xy) - mean(distances_x) - mean(distances_y)
#     return distance * n / 2
# end

# function decide_energy(hier_sample_1::emp_ppm, hier_sample_2::emp_ppm, θ::Float64, n_samples::Int)
#     n = hier_sample_1.n

#     atoms_1 = sort(hier_sample_1.atoms, dims = 2)
#     atoms_2 = sort(hier_sample_2.atoms, dims = 2)

    
#     observed_test_stat = test_statistic_energy(atoms_1, atoms_2)
        
#     # obtain quantile using bootstrap approach
#     bootstrap_samples = zeros(n_samples) # zeros can be improved
    
#     total_rows = vcat(atoms_1, atoms_2) # collect all rows
#     for i in 1:n_samples
#         indices_1 = sample(1:2*n, n; replace = true)
#         indices_2 = sample(1:2*n, n; replace = true)
    
#         bootstrap_samples[i] = test_statistic_energy(total_rows[indices_1,:], total_rows[indices_2,:])
#     end
#     threshold = quantile(bootstrap_samples, 1 - θ)
    
#     return 1.0*(observed_test_stat > threshold)
# end


# function decide_dm(mu_1::Vector{Float64}, mu_2::Vector{Float64}, θ::Float64, n_bootstrap::Int)
#     n = length(mu_1)
    
#     @rput mu_1 mu_2 n n_bootstrap
#     R"""

#     library(frechet)
#     n1 <- n
#     n2 <- n
#     delta <- 1
#     qSup <- seq(0.01, 0.99, (0.99 - 0.01) / 50)

#     Y1 <- lapply(1:n1, function(i) {
#     qnorm(qSup, mu_1[i], sd = 1)
#     })
#     Y2 <- lapply(1:n2, function(i) {
#     qnorm(qSup, mu_2[i], sd = 1)
#     })
#     Ly <- c(Y1, Y2)
#     Lx <- qSup
#     group <- c(rep(1, n1), rep(2, n2))
#     res <- DenANOVA(qin = Ly, supin = Lx, group = group, optns = list(boot = TRUE, R = n_bootstrap))
#     pvalue = res$pvalBoot # returns bootstrap pvalue
#     """
#     @rget pvalue  
#     return 1 * (pvalue < θ)
# end



# function rejection_rate_dm(q_1::Union{tnormal_normal,simple_discr_1,mixture_ppm}, 
#                 q_2::Union{tnormal_normal,simple_discr_1,mixture_ppm}, n::Int,
#              S::Int, θ::Float64, n_bootstrap::Int)
#     rate = 0.0
    
#     for i in 1:S
#         # generate normal distributions 
#         mu_1 = generate_prob_measures(q_1, n) # only contains means for normal distribution
#         mu_2 = generate_prob_measures(q_2, n) # only contains means for normal distribution
        
#         rate += decide_dm(mu_1, mu_2, θ, n_bootstrap) 
#     end
#     return rate/S
# end





# function threshold_hipm(hier_sample_1::emp_ppm, hier_sample_2::emp_ppm, θ::Float64, n_samples::Int, bootstrap::Bool)
#     n = hier_sample_1.n
#     m = hier_sample_1.m
#     # set endpoints
#     a = minimum((hier_sample_1.a, hier_sample_2.a))
#     b = maximum((hier_sample_1.b, hier_sample_2.b))
#     hier_sample_1.a = a
#     hier_sample_2.a = a
#     hier_sample_1.b = b
#     hier_sample_2.b = b

#     samples = zeros(n_samples)
#     total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
#     if bootstrap
#         for i in 1:n_samples
#             indices_1 = sample(1:2*n, n; replace = true)
#             indices_2 = sample(1:2*n, n; replace = true)

#             atoms_1 = total_rows[indices_1,:] # first rows indexed by n random indices to the atoms_1
#             atoms_2 = total_rows[indices_2,:] # first rows indexed by n random indices to the atoms_2

#             hier_sample_1_bootstrap = emp_ppm(atoms_1, n, m, a, b)
#             hier_sample_2_bootstrap = emp_ppm(atoms_2, n, m, a, b)

#             samples[i] = dlip(hier_sample_1_bootstrap, hier_sample_2_bootstrap)
#         end
#     else
#         for i in 1:n_samples
#             random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

#             atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
#             atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
#             hier_sample_1_permutation = emp_ppm(atoms_1, n, m, a, b)
#             hier_sample_2_permutation = emp_ppm(atoms_2, n, m, a, b)

#             samples[i] = dlip(hier_sample_1_permutation, hier_sample_2_permutation)
#         end
#     end
#     return quantile(samples, 1 - θ)
# end


# function threshold_wow(hier_sample_1::emp_ppm, hier_sample_2::emp_ppm, θ::Float64, n_samples::Int, bootstrap::Bool)
#     n = hier_sample_1.n

#     atoms_1 = sort(hier_sample_1.atoms, dims = 2)
#     atoms_2 = sort(hier_sample_2.atoms, dims = 2)

#     samples = zeros(n_samples)
#     total_rows = vcat(atoms_1, atoms_2) # collect all rows
#     if bootstrap
#         for i in 1:n_samples
#             indices_1 = sample(1:2*n, n; replace = true)
#             indices_2 = sample(1:2*n, n; replace = true)

#             new_atoms_1 = total_rows[indices_1,:] # first rows indexed by n random indices to the atoms_1
#             new_atoms_2 = total_rows[indices_2,:] # first rows indexed by n random indices to the atoms_2

#             samples[i] = ww(new_atoms_1, new_atoms_2, true) # sorted = true
#         end
#     else
#         for i in 1:n_samples
#             random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

#             new_atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
#             new_atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
         
#             samples[i] = ww(new_atoms_1, new_atoms_2, true) # sorted = true
#         end
#     end
#     return quantile(samples, 1 - θ)
# end






# function rejection_rate_all(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_samples::Int, bootstrap::Bool)
#     # if bootstrap is true then do bootstrap approach, n_samples refers to either number of permutations or bootstraps

#     # firstly we obtain fixed thresholds for HIPM and WoW
#     aux_hier_sample_1 = generate_emp(q_1,n,m)
#     aux_hier_sample_2 = generate_emp(q_2, n, m)
#     threshold_hipm_wrong = threshold_hipm(aux_hier_sample_1, aux_hier_sample_2, θ, n_samples, bootstrap) # gasaketebeli
#     threshold_wow_wrong = threshold_wow(aux_hier_sample_1, aux_hier_sample_2, θ, n_samples, bootstrap) # gasaketebeli

#     rates_hipm = 0.0
#     rates_wow = 0.0
#     rates_energy = 0.0

#     @floop ThreadedEx() for s in 1:S
#         # generate samples and set endpoints
#         hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
#         a = minimum((hier_sample_1.a, hier_sample_2.a))
#         b = maximum((hier_sample_1.b, hier_sample_2.b))
#         hier_sample_1.a = a
#         hier_sample_2.a = a
#         hier_sample_1.b = b
#         hier_sample_2.b = b

#         # record decisions from each testing methods
#         @reduce rates_hipm += 1.0*(dlip(hier_sample_1, hier_sample_2) > threshold_hipm_wrong)
#         @reduce rates_wow += 1.0 * (ww(hier_sample_1, hier_sample_2) > threshold_wow_wrong)
#         @reduce rates_energy += decide_energy(hier_sample_1, hier_sample_2, θ, n_samples) 
#     end
#     rates_energy /= S
#     rates_wow /= S
#     rates_hipm /= S
#     rates_dm = 0.0
#     rates_dm = rejection_rate_dm(q_1, q_2, n, S, θ, n_samples)
#     return rates_hipm,rates_wow,rates_dm,rates_energy
# end

function save_fig(pairs::Vector{<:Tuple{PPM,PPM}}, param_pairs::Vector{Float64}, file_name::String, file_path::String, title::String, xlabel::String, ylabel::String,
    n::Int, m::Int, S::Int, θ::Float64, n_samples::Int, bootstrap::Bool)
    rates_hipm = zeros(length(param_pairs))
    rates_wow = zeros(length(param_pairs))
    rates_dm = zeros(length(param_pairs))
    rates_energy = zeros(length(param_pairs))
    for i in 1:length(pairs)
        q_1, q_2 = pairs[i]
        r_hipm, r_wow, r_dm, r_energy = rejection_rate_all(q_1,q_2,n,m,S,θ,n_samples,bootstrap)
        rates_hipm[i] = r_hipm
        rates_wow[i] = r_wow
        rates_dm[i] = r_dm
        rates_energy[i] = r_energy
        println(i)
    end
    fig = plot(title = title, xlabel = xlabel, ylabel = ylabel, xlims=(minimum(param_pairs) - 0.05, maximum(param_pairs)+ 0.05),
                         ylims = (-0.1, 1.1))
    plot!(fig, param_pairs, rates_dm, label = "dm", color = "red", marker = (:circle, 4))
    plot!(fig, param_pairs, rates_hipm, label = "hipm", color = "green", marker = (:circle, 4))
    plot!(fig, param_pairs, rates_wow, label = "wow", color = "brown", marker = (:circle, 4))
    plot!(fig, param_pairs, rates_energy, label = "Energy", color = "blue", marker = (:circle, 4))
    filepath = joinpath(pwd(), file_path)
    savefig(fig,joinpath(filepath, file_name))
end


# obtain times: for S = 4, 1 pair of RPMS, n = 100, m = 100 and n_samples = 100.

# wow : 23 seconds (40 minutes in total if proper threshold, S = 400)
#        70 seconds in total if wrong threshold, S = 400
        
# hipm : 50 seconds ( 83 minutes in total if proper threshold, S = 400)
#         210 seconds in total if wrong threshold, S = 400    

# energy: 4 seconds 
#           400 seconds in total, S = 400

# dm : 0.7 seconds ( S = 1 )
#      280 seconds in total, S = 400
        
# fig 1 with varying mean

# δs = collect(-1.0:0.1:1.0)
# pairs = [(tnormal_normal(0.0,0.5,-10.0,10.0), tnormal_normal(δ, 0.5, -10.0,10.0)) for δ in δs]
# file_path = "plots/frechet/figure1"
# title = "Rejection rates for 4 schemes"
# xlabel = "δ"
# ylabel = "Rej rate"
# n = 100
# m = 100
# S = 12
# n_samples = 100
# θ = 0.05
# bootstrap = true
# file_name = "varying_mean_n=$(n)_m=$(m)_S=$(S)_bootstrap_n_samples=$(n_samples)"
# t = time()
# save_fig(pairs, δs, file_name, file_path, title, xlabel,ylabel, n,m,S,θ,n_samples,bootstrap)
# dur = time() - t





# fig 1 with varying variance
# τs = collect(0.1:0.1:3.0)
# pairs = [(tnormal_normal(0.0,0.2,-10.0,10.0), tnormal_normal(0.0,0.2*τ,-10.0,10.0)) for τ in τs]
# file_path = "plots/frechet/figure1"
# title = "Rejection rates for 4 schemes"
# xlabel = "τ"
# ylabel = "Rej rate"
# n = 10
# m = 10
# S = 1
# n_samples = 1
# θ = 0.05
# bootstrap = true
# file_name = "varying_variance_n=$(n)_m=$(m)_S=$(S)_bootstrap_n_samples=$(n_samples)"
# t = time()
# save_fig(pairs, τs, file_name, file_path, title, xlabel,ylabel, n,m,S,θ,n_samples,bootstrap)
# dur = time() - t

# counterexample 

# λs = collect(0.0:0.1:1.0)
# q_1 = simple_discr_1()
# q_2_aux = simple_discr_2()
# pairs = [(q_1, mixture_ppm(q_1, q_2_aux, λ)) for λ in λs]
# file_path = "plots/frechet/counterexample"
# title = "Rejection rates for 4 schemes"
# xlabel = "λ"
# ylabel = "Rej rate"
# n = 100
# m = 100
# S = 4
# n_samples = 100
# θ = 0.05
# bootstrap = true
# file_name = "counterexample_n=$(n)_m=$(m)_S=$(S)_bootstrap_n_samples=$(n_samples)"
# t = time()
# save_fig(pairs, λs, file_name, file_path, title, xlabel,ylabel, n,m,S,θ,n_samples,bootstrap)
# dur = time() - t