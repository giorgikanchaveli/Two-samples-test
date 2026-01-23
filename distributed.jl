# This file is for distributed computing rather than using Threads. Not using. 



# Improvements: 
#           1) generate sorted hierarchical samples.
#           2) Ignore energy and dm to check smoothness
#           3) better way to run simulations (array of jobs???)
using Distributed
using Plots


@everywhere using RCall # to call R functions
@everywhere using FLoops # for parallel computing

@everywhere include("distributions.jl")

@everywhere include("distances/new_distance.jl")
@everywhere include("distances/distance_Wasserstein.jl")

@everywhere function test_statistic_energy(atoms_1::Matrix{Float64}, atoms_2::Matrix{Float64})
    # we assume that rows in each atoms are sorted.
    n = size(atoms_1)[1]
    
    distances_x = Matrix{Float64}(undef, n, n)
    distances_xy = Matrix{Float64}(undef, n, n)
    distances_y = Matrix{Float64}(undef, n, n)

    for i in 1:n
        x = atoms_1[i,:]
        y = atoms_2[i,:]
        for j in 1:n
            distances_x[i, j] = wasserstein1DUniform_sorted(x, atoms_1[j,:], 1)
            distances_xy[i, j] = wasserstein1DUniform_sorted(x, atoms_2[j,:], 1)
            distances_y[i, j] = wasserstein1DUniform_sorted(y, atoms_2[j,:], 1)
        end
    end
    distance = 2 * mean(distances_xy) - mean(distances_x) - mean(distances_y)
    return distance * n / 2
end

# @everywhere function threshold_energy(hier_sample_1::HierSample, hier_sample_2::HierSample, θ::Float64, n_samples::Int)
#     n = hier_sample_1.n
#     atoms_1 = sort(hier_sample_1.atoms, dims = 2)
#     atoms_2 = sort(hier_sample_2.atoms, dims = 2)

#     # obtain quantile using bootstrap approach
#     bootstrap_samples = zeros(n_samples) # zeros can be improved
    
#     total_rows = vcat(atoms_1, atoms_2) # collect all rows
#     @floop ThreadedEx() for i in 1:n_samples
#         indices_1 = sample(1:2*n, n; replace = true)
#         indices_2 = sample(1:2*n, n; replace = true)
    
#         bootstrap_samples[i] = test_statistic_energy(total_rows[indices_1,:], total_rows[indices_2,:])
#     end
#     return quantile(bootstrap_samples, 1 - θ)
# end


@everywhere function threshold_energy_chunk(total_rows::Matrix{Float64},
                                            n::Int,
                                            n_samples_chunk::Int)
    bootstrap_samples = zeros(n_samples_chunk)

    @floop ThreadedEx() for i in 1:n_samples_chunk
        indices_1 = sample(1:2n, n; replace = true)
        indices_2 = sample(1:2n, n; replace = true)

        bootstrap_samples[i] =test_statistic_energy(total_rows[indices_1, :],
                                  total_rows[indices_2, :])
    end
    return bootstrap_samples
end

function threshold_energy(hier_sample_1::HierSample,hier_sample_2::HierSample,θ::Float64, n_samples::Int)

    n = hier_sample_1.n
    atoms_1 = sort(hier_sample_1.atoms, dims = 2)
    atoms_2 = sort(hier_sample_2.atoms, dims = 2)
    total_rows = vcat(atoms_1, atoms_2)

    num_w = nworkers()
    @assert n_samples % num_w == 0
    n_samples_chunk = n_samples ÷ num_w

    # Each worker produces a Vector{Float64} of length n_samples_chunk
    partials = pmap(1:num_w) do _
        threshold_energy_chunk(total_rows, n, n_samples_chunk)
    end

    # Concatenate into a single Vector{Float64} of length n_samples
    bootstrap_samples = vcat(partials...)

    return quantile(bootstrap_samples, 1 - θ)
end




@everywhere function decide_dm(mu_1::Vector{Float64}, mu_2::Vector{Float64}, θ::Float64, n_bootstrap::Int)
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



@everywhere function rejection_rate_dm(q_1::Union{tnormal_normal,simple_discr_1, simple_discr_2,mixture_ppm},
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

@everywhere function threshold_hipm_chunk(total_rows::Matrix{Float64},
                                          n::Int,
                                          m::Int,
                                          a::Float64,
                                          b::Float64,
                                          n_samples_chunk::Int,
                                          bootstrap::Bool)
    samples = zeros(n_samples_chunk)

    if bootstrap
        @floop ThreadedEx() for i in 1:n_samples_chunk
            indices_1 = sample(1:2n, n; replace = true)
            indices_2 = sample(1:2n, n; replace = true)

            atoms_1 = total_rows[indices_1, :]      # n×m
            atoms_2 = total_rows[indices_2, :]      # n×m

            hier_1 = HierSample(atoms_1, n, m, a, b)
            hier_2 = HierSample(atoms_2, n, m, a, b)

            samples[i] = dlip(hier_1, hier_2)
        end
    else
        @floop ThreadedEx() for i in 1:n_samples_chunk
            random_indices = randperm(2n)

            atoms_1 = total_rows[random_indices[1:n],     :]
            atoms_2 = total_rows[random_indices[n+1:end], :]

            hier_1 = HierSample(atoms_1, n, m, a, b)
            hier_2 = HierSample(atoms_2, n, m, a, b)

            samples[i] = dlip(hier_1, hier_2)
        end
    end

    return samples
end


function threshold_hipm(hier_sample_1::HierSample,
                        hier_sample_2::HierSample,
                        θ::Float64,
                        n_samples::Int,
                        bootstrap::Bool)
    n = hier_sample_1.n
    m = hier_sample_1.m

    # Set endpoints
    a = min(hier_sample_1.a, hier_sample_2.a)
    b = max(hier_sample_1.b, hier_sample_2.b)
    hier_sample_1.a = a
    hier_sample_2.a = a
    hier_sample_1.b = b
    hier_sample_2.b = b

    # Stack atoms once on the master
    total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms)

    num_w = nworkers()
    @assert n_samples % num_w == 0
    n_samples_chunk = n_samples ÷ num_w

    # Each worker produces a Vector{Float64} of length n_samples_chunk
    partials = pmap(1:num_w) do _
        threshold_hipm_chunk(total_rows, n, m, a, b, n_samples_chunk, bootstrap)
    end

    # Concatenate all bootstrap samples
    samples = vcat(partials...)

    return quantile(samples, 1 - θ)
end





# @everywhere function threshold_hipm(hier_sample_1::HierSample, hier_sample_2::HierSample, θ::Float64, n_samples::Int, bootstrap::Bool)
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
#         @floop ThreadedEx() for i in 1:n_samples
#             indices_1 = sample(1:2*n, n; replace = true)
#             indices_2 = sample(1:2*n, n; replace = true)

#             atoms_1 = total_rows[indices_1,:] # first rows indexed by n random indices to the atoms_1
#             atoms_2 = total_rows[indices_2,:] # first rows indexed by n random indices to the atoms_2

#             hier_sample_1_bootstrap = HierSample(atoms_1, n, m, a, b)
#             hier_sample_2_bootstrap = HierSample(atoms_2, n, m, a, b)

#             samples[i] = dlip(hier_sample_1_bootstrap, hier_sample_2_bootstrap)
           
#         end
#     else
#         @floop ThreadedEx() for i in 1:n_samples
#             random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

#             atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
#             atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
#             hier_sample_1_permutation = HierSample(atoms_1, n, m, a, b)
#             hier_sample_2_permutation = HierSample(atoms_2, n, m, a, b)
#             samples[i] = dlip(hier_sample_1_permutation, hier_sample_2_permutation)
            
#         end
#     end
#     return quantile(samples, 1 - θ)
# end


# @everywhere function threshold_wow(hier_sample_1::HierSample, hier_sample_2::HierSample, θ::Float64, n_samples::Int, bootstrap::Bool)
#     n = hier_sample_1.n

#     atoms_1 = sort(hier_sample_1.atoms, dims = 2)
#     atoms_2 = sort(hier_sample_2.atoms, dims = 2)

#     samples = zeros(n_samples)
#     total_rows = vcat(atoms_1, atoms_2) # collect all rows
#     if bootstrap
#         @floop ThreadedEx() for i in 1:n_samples
#             indices_1 = sample(1:2*n, n; replace = true)
#             indices_2 = sample(1:2*n, n; replace = true)

#             new_atoms_1 = total_rows[indices_1,:] # first rows indexed by n random indices to the atoms_1
#             new_atoms_2 = total_rows[indices_2,:] # first rows indexed by n random indices to the atoms_2
#             samples[i] = ww(new_atoms_1, new_atoms_2, true)
            
#         end
#     else
#         @floop ThreadedEx() for i in 1:n_samples
#             random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

#             new_atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
#             new_atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
#             samples[i] = ww(new_atoms_1, new_atoms_2, true)
            
#         end
#     end
#     return quantile(samples, 1 - θ)
# end
@everywhere function threshold_wow_chunk(total_rows::Matrix{Float64},
                                         n::Int,
                                         n_samples_chunk::Int,
                                         bootstrap::Bool)
    samples = zeros(n_samples_chunk)

    if bootstrap
        @floop ThreadedEx() for i in 1:n_samples_chunk
            indices_1 = sample(1:2n, n; replace = true)
            indices_2 = sample(1:2n, n; replace = true)

            new_atoms_1 = total_rows[indices_1, :]
            new_atoms_2 = total_rows[indices_2, :]

            samples[i] = ww(new_atoms_1, new_atoms_2, true)  # sorted = true
        end
    else
        @floop ThreadedEx() for i in 1:n_samples_chunk
            random_indices = randperm(2n)

            new_atoms_1 = total_rows[random_indices[1:n],     :]
            new_atoms_2 = total_rows[random_indices[n+1:end], :]

            samples[i] = ww(new_atoms_1, new_atoms_2, true)  # sorted = true
        end
    end

    return samples
end
function threshold_wow(hier_sample_1::HierSample,
                       hier_sample_2::HierSample,
                       θ::Float64,
                       n_samples::Int,
                       bootstrap::Bool)
    n = hier_sample_1.n

    atoms_1 = sort(hier_sample_1.atoms, dims = 2)
    atoms_2 = sort(hier_sample_2.atoms, dims = 2)
    total_rows = vcat(atoms_1, atoms_2)

    num_w = nworkers()
    @assert n_samples % num_w == 0
    n_samples_chunk = n_samples ÷ num_w

    # Each worker produces a Vector{Float64} of length n_samples_chunk
    partials = pmap(1:num_w) do _
        threshold_wow_chunk(total_rows, n, n_samples_chunk, bootstrap)
    end

    # Concatenate all bootstrap samples
    samples = vcat(partials...)

    return quantile(samples, 1 - θ)
end








@everywhere function rejection_rate_all_chunk(q_1::PPM, q_2::PPM, n::Int, m::Int, S_chunk::Int,
                     θ::Float64, n_samples::Int, threshold_hipm_wrong::Float64, threshold_wow_wrong::Float64, threshold_energy_wrong::Float64)
    # if bootstrap is true then do bootstrap approach, n_samples refers to either number of permutations or bootstraps


    rates_hipm = 0.0
    rates_wow = 0.0
    rates_energy = 0.0

    @floop ThreadedEx() for s in 1:S_chunk
        # generate samples and set endpoints
        hier_sample_1, hier_sample_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
        a = minimum((hier_sample_1.a, hier_sample_2.a))
        b = maximum((hier_sample_1.b, hier_sample_2.b))
        hier_sample_1.a = a
        hier_sample_2.a = a
        hier_sample_1.b = b
        hier_sample_2.b = b

        # record decisions from each testing methods
        @reduce rates_hipm += 1.0*(dlip(hier_sample_1, hier_sample_2) > threshold_hipm_wrong)
        @reduce rates_wow += 1.0 * (ww(hier_sample_1, hier_sample_2) > threshold_wow_wrong)
        # sort atoms before energy statistic
        atoms_1 = sort(hier_sample_1.atoms, dims = 2)
        atoms_2 = sort(hier_sample_2.atoms, dims = 2)
        @reduce rates_energy += 1.0 * (test_statistic_energy(atoms_1, atoms_2) > threshold_energy_wrong)
    end
    rates_energy /= S_chunk
    rates_wow /= S_chunk
    rates_hipm /= S_chunk
    rates_dm = rejection_rate_dm(q_1, q_2, n, S_chunk, θ, n_samples)
    return rates_hipm, rates_wow, rates_energy, rates_dm
end


function rejection_rate_all(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int,
                     θ::Float64, n_samples::Int, bootstrap::Bool)
    
    # firstly we obtain fixed thresholds for HIPM and WoW
    aux_hier_sample_1 = generate_hiersample(q_1,n,m)
    aux_hier_sample_2 = generate_hiersample(q_2, n, m)
    threshold_hipm_wrong = threshold_hipm(aux_hier_sample_1, aux_hier_sample_2, θ, n_samples, bootstrap) # gasaketebeli
    threshold_wow_wrong = threshold_wow(aux_hier_sample_1, aux_hier_sample_2, θ, n_samples, bootstrap) # gasaketebeli
    threshold_energy_wrong = threshold_energy(aux_hier_sample_1, aux_hier_sample_2,
                                θ, n_samples)

    num_w = nworkers()
    @assert S % num_w == 0
    S_chunk = S ÷ num_w

    partials = pmap(1:num_w) do _
            rejection_rate_all_chunk(q_1, q_2, n, m, S_chunk, θ, n_samples, 
                threshold_hipm_wrong, threshold_wow_wrong, threshold_energy_wrong)
    end

    rates_hipm   = sum(t[1] for t in partials) / num_w
    rates_wow    = sum(t[2] for t in partials) / num_w
    rates_energy = sum(t[3] for t in partials) / num_w
    rates_dm = sum(t[4] for t in partials) / num_w
    return rates_hipm, rates_wow, rates_dm, rates_energy
end


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


println("number of workers is : $(nworkers())")
println("number of threads per worker: $(Threads.nthreads())")




println("running file distributed.jl")

println("expected duration is x hours")

τs = collect(0.1:0.05:3.0)
#τs = [1.0]

pairs = [(tnormal_normal(0.0,0.2,-10.0,10.0), tnormal_normal(0.0,0.2*τ,-10.0,10.0)) for τ in τs]
file_path = "plots/frechet/figure1"
title = "Rejection rates for 4 schemes"
xlabel = "τ"
ylabel = "Rej rate"
n = 100
m = 200
S = 1000
n_samples = 1000
θ = 0.05
bootstrap = false
file_name = "varying_variance_n=$(n)_m=$(m)_S=$(S)_permutation_n_samples=$(n_samples)"
t = time()
save_fig(pairs, τs, file_name, file_path, title, xlabel,ylabel, n,m,S,θ,n_samples,bootstrap)
dur = time() - t
println("total duration is $(dur/3600) hours")
