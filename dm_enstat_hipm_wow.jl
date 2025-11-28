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






# q_1 = tnormal_normal(0.0,0.5,-10.0,10.0)
# q_2 = tnormal_normal(0.3,0.5,-10.0,10.0)
# t = time()
# rates_hipm,rates_wow,rates_dm,rates_energy = rejection_rate_all(q_1,q_2,100,
#             100,50,0.05,100,true)
# dur = time() - t
# rates_dm
# rates_energy







# function save_varying_mean_boostrap(n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)
#     δs = collect(-1.0:2.0:1.0)

#     rej_rates_hipm = zeros(length(δs))
#     rej_rates_wow = zeros(length(δs))
#     rej_rates_dm = zeros(length(δs))
#     rej_rates_energy = zeros(length(δs))

#     for (i, δ) in enumerate(δs)
#         μ_1, σ_1, a, b = 0.0, 0.5, -10.0, 10.0
#         μ_2, σ_2, a, b = δ, 0.5, -10.0, 10.0

#         q_1 = tnormal_normal(μ_1, σ_1, a, b)
#         q_2 = tnormal_normal(μ_2, σ_2, a, b)

#         rej_rates_hipm[i] = rejection_rate_hipm_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
#         rej_rates_dm[i] = rejection_rate_dm_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
#         rej_rates_wow[i] = rejection_rate_wow_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
#         rej_rates_energy[i] = rejection_rate_energy_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
#     end

#     varying_mean_boostrap = plot(title = "Rejection rates of 4 testing schemes, boostrap", xlabel = "δ", ylabel = "Rej rate", xlims=(-1.0, 1.1), ylims = (-0.1, 1.1))
#     plot!(varying_mean_boostrap, δs, rej_rates_dm, label = "dm", color = "red",marker = (:circle, 4))
#     plot!(varying_mean_boostrap, δs, rej_rates_hipm, label = "hipm", color = "green", marker = (:circle, 4))
#     plot!(varying_mean_boostrap, δs, rej_rates_wow, label = "wow", color = "brown", marker = (:circle, 4))
#     plot!(varying_mean_boostrap, δs, rej_rates_energy, label = "Energy", color = "blue", marker = (:circle, 4))
#     filepath = joinpath(pwd(), "frechet/figure1")
#     savefig(varying_mean_boostrap,joinpath(filepath, "varying_mean_n=$(n)_m=$(m)_S=$(S)_nboostrap=$(n_boostrap).png"))
# end




# function save_varying_mean_permutation(n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)
#     # note that we still use boostrap for energy and dm
#     δs = collect(-1.0:0.1:1.0)

#     rej_rates_hipm = zeros(length(δs))
#     rej_rates_wow = zeros(length(δs))
#     rej_rates_dm = zeros(length(δs))
#     rej_rates_energy = zeros(length(δs))

#     μ_1, σ_1, a, b = 0.0, 0.5, -10.0, 10.0
#     q_1 = tnormal_normal(μ_1, σ_1, a, b)

#     for (i, δ) in enumerate(δs)
#         q_2 = tnormal_normal(δ, σ_1, a, b)

#         rej_rates_hipm[i] = rejection_rate_hipm_permutation_wrong(q_1, q_2, n, m, S, θ, n_permutation)
#         rej_rates_dm[i] = rejection_rate_dm_boostrap_parallel(q_1, q_2, n, m, S, θ, n_permutation)
#         rej_rates_wow[i] = rejection_rate_wow_permutation_wrong(q_1, q_2, n, m, S, θ, n_permutation)
#         rej_rates_energy[i] = rejection_rate_energy_boostrap_parallel(q_1, q_2, n, m, S, θ, n_permutation)
#     end

#     varying_mean_permutation = plot(title = "Rejection rates of 4 testing schemes, permutation", xlabel = "δ", ylabel = "Rej rate", xlims=(-1.0, 1.1), ylims = (-0.1, 1.1))
#     plot!(varying_mean_permutation, δs, rej_rates_dm, label = "dm", color = "red", marker = (:circle, 4))
#     plot!(varying_mean_permutation, δs, rej_rates_hipm, label = "hipm", color = "green", marker = (:circle, 4))
#     plot!(varying_mean_permutation, δs, rej_rates_wow, label = "wow", color = "brown", marker = (:circle, 4))
#     plot!(varying_mean_permutation, δs, rej_rates_energy, label = "Energy", color = "blue", marker = (:circle, 4))
#     filepath = joinpath(pwd(), "frechet/figure1")
#     savefig(varying_mean_permutation,joinpath(filepath, "varying_mean_n=$(n)_m=$(m)_S=$(S)_npermutation=$(n_permutation).png"))
# end


# function save_varying_variance_boostrap(n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)
#     τs = collect(0.1:0.1:3.0)

#     rej_rates_hipm = zeros(length(τs))
#     rej_rates_wow = zeros(length(τs))
#     rej_rates_dm = zeros(length(τs))
#     rej_rates_energy = zeros(length(τs))

#     for (i, τ) in enumerate(τs)
#         μ_1, σ_1, a, b = 0.0, 0.2, -10.0, 10.0

#         q_1 = tnormal_normal(μ_1, σ_1, a, b)
#         q_2 = tnormal_normal(μ_1, σ_1*τ, a, b)

#         rej_rates_hipm[i] = rejection_rate_hipm_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
#         rej_rates_dm[i] = rejection_rate_dm_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
#         rej_rates_wow[i] = rejection_rate_wow_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
#         rej_rates_energy[i] = rejection_rate_energy_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
#     end

#     varying_variance_boostrap = plot(title = "Rejection rates of 4 testing schemes, boostrap", xlabel = "τ", ylabel = "Rej rate", 
#                                         xlims=(0.0, 3.1), ylims = (-0.1, 1.1))
#     plot!(varying_variance_boostrap, τs, rej_rates_dm, label = "dm", color = "red",marker = (:circle, 4))
#     plot!(varying_variance_boostrap, τs, rej_rates_hipm, label = "hipm", color = "green", marker = (:circle, 4))
#     plot!(varying_variance_boostrap, τs, rej_rates_wow, label = "wow", color = "brown", marker = (:circle, 4))
#     plot!(varying_variance_boostrap, τs, rej_rates_energy, label = "Energy", color = "blue", marker = (:circle, 4))
#     filepath = joinpath(pwd(), "frechet/figure1")
#     savefig(varying_variance_boostrap,joinpath(filepath, "varying_variance_n=$(n)_m=$(m)_S=$(S)_nboostrap=$(n_boostrap).png"))
# end

# function save_varying_variance_permutation(n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)
#     τs = collect(0.1:0.1:3.0)

#     rej_rates_hipm = zeros(length(τs))
#     rej_rates_wow = zeros(length(τs))
#     rej_rates_dm = zeros(length(τs))
#     rej_rates_energy = zeros(length(τs))

#     for (i, τ) in enumerate(τs)
#         μ_1, σ_1, a, b = 0.0, 0.2, -10.0, 10.0

#         q_1 = tnormal_normal(μ_1, σ_1, a, b)
#         q_2 = tnormal_normal(μ_1, σ_1*τ, a, b)

#         rej_rates_hipm[i] = rejection_rate_hipm_permutation_wrong(q_1, q_2, n, m, S, θ, n_permutation)
#         rej_rates_dm[i] = rejection_rate_dm_boostrap_parallel(q_1, q_2, n, m, S, θ, n_permutation)
#         rej_rates_wow[i] = rejection_rate_wow_permutation_wrong(q_1, q_2, n, m, S, θ, n_permutation)
#         rej_rates_energy[i] = rejection_rate_energy_boostrap_parallel(q_1, q_2, n, m, S, θ, n_permutation)
#     end

#     varying_variance_permutation = plot(title = "Rejection rates of 4 testing schemes, permutation", xlabel = "τ", ylabel = "Rej rate", 
#                                         xlims=(0.0, 3.1), ylims = (-0.1, 1.1))
#     plot!(varying_variance_permutation, τs, rej_rates_dm, label = "dm", color = "red",marker = (:circle, 4))
#     plot!(varying_variance_permutation, τs, rej_rates_hipm, label = "hipm", color = "green", marker = (:circle, 4))
#     plot!(varying_variance_permutation, τs, rej_rates_wow, label = "wow", color = "brown", marker = (:circle, 4))
#     plot!(varying_variance_permutation, τs, rej_rates_energy, label = "Energy", color = "blue", marker = (:circle, 4))
#     filepath = joinpath(pwd(), "frechet/figure1")
#     savefig(varying_variance_permutation,joinpath(filepath, "varying_variance_n=$(n)_m=$(m)_S=$(S)_npermutation=$(n_permutation).png"))
# end

# function save_counterexample_boostrap(n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)

#     λs = collect(0.0:0.1:1.0)

#     rej_rates_hipm = zeros(length(λs))
#     rej_rates_wow = zeros(length(λs))
#     rej_rates_dm = zeros(length(λs))
#     rej_rates_energy = zeros(length(λs))

#     q_1 = simple_discr_1()
#     q_2_aux = simple_discr_2()

#     for (i, λ) in enumerate(λs)
#         q_2 = mixture_ppm(q_1, q_2_aux, λ)

#         rej_rates_hipm[i] = rejection_rate_hipm_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
#         rej_rates_dm[i] = rejection_rate_dm_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
#         rej_rates_wow[i] = rejection_rate_wow_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
#         rej_rates_energy[i] = rejection_rate_energy_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
#     end
#     counterexmaple_boostrap = plot(title = "Rejection rates of 4 testing schemes, boostrap", xlabel = "λ", ylabel = "Rej rate", 
#                                     xlims=(-0.1, 1.1), ylims = (-0.1, 1.1))
#     plot!(counterexmaple_boostrap, λs, rej_rates_dm, label = "dm", color = "red",marker = (:circle, 4))
#     plot!(counterexmaple_boostrap, λs, rej_rates_hipm, label = "hipm", color = "green", marker = (:circle, 4))
#     plot!(counterexmaple_boostrap, λs, rej_rates_wow, label = "wow", color = "brown", marker = (:circle, 4))
#     plot!(counterexmaple_boostrap, λs, rej_rates_energy, label = "Energy", color = "blue", marker = (:circle, 4))
#     filepath = joinpath(pwd(), "frechet/counterexample")
#     savefig(counterexmaple_boostrap,joinpath(filepath, "counterexample_boostrap_n=$(n)_m=$(m)_S=$(S)_nboostrap=$(n_boostrap).png"))
# end


# function save_counterexample_permutation(n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)

#     λs = collect(0.0:0.1:1.0)

#     rej_rates_hipm = zeros(length(λs))
#     rej_rates_wow = zeros(length(λs))
#     rej_rates_dm = zeros(length(λs))
#     rej_rates_energy = zeros(length(λs))

#     q_1 = simple_discr_1()
#     q_2_aux = simple_discr_2()

#     for (i, λ) in enumerate(λs)
#         q_2 = mixture_ppm(q_1, q_2_aux, λ)

#         rej_rates_hipm[i] = rejection_rate_hipm_permutation_wrong(q_1, q_2, n, m, S, θ, n_permutation)
#         rej_rates_dm[i] = rejection_rate_dm_boostrap_parallel(q_1, q_2, n, m, S, θ, n_permutation)
#         rej_rates_wow[i] = rejection_rate_wow_permutation_wrong(q_1, q_2, n, m, S, θ, n_permutation)
#         rej_rates_energy[i] = rejection_rate_energy_boostrap_parallel(q_1, q_2, n, m, S, θ, n_permutation)
#     end
#     counterexmaple_permutation = plot(title = "Rejection rates of 4 testing schemes, permutation", xlabel = "λ", ylabel = "Rej rate", 
#                                     xlims=(-0.1, 1.1), ylims = (-0.1, 1.1))
#     plot!(counterexmaple_permutation, λs, rej_rates_dm, label = "dm", color = "red",marker = (:circle, 4))
#     plot!(counterexmaple_permutation, λs, rej_rates_hipm, label = "hipm", color = "green", marker = (:circle, 4))
#     plot!(counterexmaple_permutation, λs, rej_rates_wow, label = "wow", color = "brown", marker = (:circle, 4))
#     plot!(counterexmaple_permutation, λs, rej_rates_energy, label = "Energy", color = "blue", marker = (:circle, 4))
#     filepath = joinpath(pwd(), "frechet/counterexample")
#     savefig(counterexmaple_permutation,joinpath(filepath, "counterexample_permutation_n=$(n)_m=$(m)_S=$(S)_npermutation=$(n_permutation).png"))
# end





# function save_counterexample_boostrap_only_dm(n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)

#     λs = collect(0.0:0.1:1.0)

#     rej_rates_dm = zeros(length(λs))
   

#     q_1 = simple_discr_1()
#     q_2_aux = simple_discr_2()

#     for (i, λ) in enumerate(λs)
#         q_2 = mixture_ppm(q_1, q_2_aux, λ)

#         rej_rates_dm[i] = rejection_rate_dm_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
#     end
#     counterexmaple_boostrap = plot(title = "Rejection rate of dm", xlabel = "λ", ylabel = "Rej rate", 
#                                     xlims=(-0.1, 1.1), ylims = (-0.1, 1.1))
#     plot!(counterexmaple_boostrap, λs, rej_rates_dm, label = "dm", color = "red",marker = (:circle, 4))
#     filepath = joinpath(pwd(), "frechet/counterexample")
#     savefig(counterexmaple_boostrap,joinpath(filepath, "dm_countexample_n=$(n)_m=$(m)_S=$(S)_nboostrap=$(n_boostrap).png"))
# end




#obtain times

# q_1 = tnormal_normal(1.0, 1.0, -10.0, 10.0)
# q_2 = tnormal_normal(1.0, 1.10, -10.0, 10.0)


# n = 100
# m = 100
# S = 2
# n_boostrap = 100
# θ = 0.05
# times = Dict()

# t = time()
# rej_rate = rejection_rate_dm_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
# t = time() - t
# times["dm"] = t / S

# t = time()
# rej_rate = rejection_rate_energy_boostrap(q_1, q_2, n, m, S, θ, n_boostrap)
# t = time() - t
# times["energy"] = t / S

# t = time()
# rej_rate_hipm = rejection_rate_hipm_permutation_parallel(q_1, q_2, n, m, S, θ, n_boostrap)
# t = time() - t
# times["hipm"] = t / S
  
# t = time()
# rej_rate_wow = rejection_rate_wow_permutation_wrong(q_1, q_2, n, m, S, θ, n_boostrap)
# t = time() - t
# times["wow"] = t / S

# for each pair q_1, q_2

#   dm : if S = 1 it takes 9 seconds, so in total 1.2 hours for S = 500
#   wow : if S = 1 it takes 31 seconds, so in total 4.3 hours for S = 500
#   energy : if S = 1 it takes 16 seconds, so in total 2.2 hours for S = 500
#   hipm : if S = 1 it takes 161 seconds, so in total 22.2 hours for S = 500




# println(Threads.nthreads())


# # Reproduce figures from the paper Dubbey & Muller
# n, m = 100, 100
# S = 416
# θ = 0.05
# n_boostrap = 100
# n_permutation = n_boostrap

# t = time()
#save_varying_mean_boostrap(n,m,S,θ,n_boostrap)
# save_varying_mean_permutation(n,m,S,θ,n_permutation)

# #save_varying_variance_boostrap(n,m,S,θ,n_boostrap)
# save_varying_variance_permutation(n,m,S,θ,n_permutation)
# duration = time() - t
# println("duration to plot figure 1: $(duration)")
# # Example where method using Frechet mean and variance fails. 


# n = 100
# m = 100
# S = 416
# θ = 0.05
# n_boostrap = 100

# t = time()
# #save_counterexample_boostrap(n,m,S,θ,n_boostrap)
# save_counterexample_permutation(n,m,S,θ,n_boostrap)
# duration = time() - t
# println("duration to plot counterexample: $(duration)")



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