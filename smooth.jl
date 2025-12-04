#include("methods.jl")
using Plots
using FLoops # for parallel computing

include("distributions.jl")

include("distances/new_distance.jl")
include("distances/distance_Wasserstein.jl")


function threshold_wow_nothread(hier_sample_1::emp_ppm, hier_sample_2::emp_ppm, θ::Float64, n_samples::Int, bootstrap::Bool)
    n = hier_sample_1.n
    atoms_1 = hier_sample_1.atoms
    atoms_2 = hier_sample_2.atoms
    
    samples = zeros(n_samples)
    total_rows = vcat(atoms_1, atoms_2) # collect all rows
    if bootstrap
        for i in 1:n_samples
            indices_1 = sample(1:2*n, n; replace = true)
            indices_2 = sample(1:2*n, n; replace = true)

            new_atoms_1 = total_rows[indices_1,:] # first rows indexed by n random indices to the atoms_1
            new_atoms_2 = total_rows[indices_2,:] # first rows indexed by n random indices to the atoms_2

            samples[i] = ww(new_atoms_1, new_atoms_2) # sorted = true
        end
    else
        for i in 1:n_samples
            random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

            new_atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
            new_atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
         
            samples[i] = ww(new_atoms_1, new_atoms_2) # sorted = true
        end
    end
    return quantile(samples, 1 - θ)
end

function rejection_rate_wow_correct(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int,
                     θ::Float64, n_samples::Int, bootstrap::Bool)
    # if bootstrap is true then do bootstrap approach, n_samples refers to either number of permutations or bootstraps



    rates_wow = 0.0
  
    @floop ThreadedEx() for s in 1:S
        # generate samples and set endpoints
        hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        
        threshold = threshold_wow_nothread(hier_sample_1, hier_sample_2, θ, n_samples, bootstrap)
        # record decisions from each testing methods
        @reduce rates_wow += 1.0 * (ww(hier_sample_1, hier_sample_2) > threshold)
    end
    rates_wow /= S
    return rates_wow
end




function save_fig(pairs::Vector{<:Tuple{PPM,PPM}}, param_pairs::Vector{Float64}, file_name::String, file_path::String, title::String, xlabel::String, ylabel::String,
    n::Int, m::Int, S::Int, θ::Float64, n_samples::Int, bootstrap::Bool)

    rates_wow = zeros(length(param_pairs))
    
    for i in 1:length(pairs)
        q_1, q_2 = pairs[i]
        rates_wow[i] = rejection_rate_wow_correct(q_1,q_2,n,m,S,θ,n_samples,bootstrap)
        println(i)
    end
    fig = plot(title = title, xlabel = xlabel, ylabel = ylabel, xlims=(minimum(param_pairs) - 0.05, maximum(param_pairs)+ 0.05),
                         ylims = (-0.1, 1.1))
    plot!(fig, param_pairs, rates_wow, label = "wow", color = "brown", marker = (:circle, 4))
    filepath = joinpath(pwd(), file_path)
    savefig(fig,joinpath(filepath, file_name))
end




println("running file smooth.jl")
println("number of threads: $(Threads.nthreads())")
println("expected duration is 3.4 hours")





# #τs = collect(0.1:0.05:3.0)
τs = collect(0.5:0.05:1.8)
pairs = [(tnormal_normal(0.0,0.2,-10.0,10.0), tnormal_normal(0.0,0.2*τ,-10.0,10.0)) for τ in τs]
file_path = "plotscluster/"
title = "Rejection rates for wow"
xlabel = "τ"
ylabel = "Rej rate"
n = 100
m = 200
S = 100
n_samples = 100
θ = 0.05
bootstrap = false
file_name = "varying_variance_truethresh_n=$(n)_m=$(m)_S=$(S)_permutation_n_samples=$(n_samples)"
t = time()
save_fig(pairs, τs, file_name, file_path, title, xlabel,ylabel, n,m,S,θ,n_samples,bootstrap)
dur = time() - t
println("total duration is $(dur/3600) hours")
println("parameters are n = $(n), m = $(m), S = $(S), n_samples = $(n_samples), permutation")


# compare thresholds: 

# q_1 = tnormal_normal(0.0,0.2,-10.0,10.0)
# τ = 1.4
# q_2 = tnormal_normal(0.0,0.2*τ,-10.0,10.0)




# n = 100
# m = 200
# n_samples = 100
# θ = 0.05

# bootstrap = true
# K = 100
# thresholds = zeros(K)
# distances = zeros(K)
# t = time()
# for i in 1:K
#     aux_hier_sample_1 = generate_emp(q_1, n, m)
#     aux_hier_sample_2 = generate_emp(q_2, n, m)
#     distances[i] = ww(aux_hier_sample_1,aux_hier_sample_2)
#     thresholds[i] = threshold_wow(aux_hier_sample_1, aux_hier_sample_2, θ, n_samples, bootstrap) # gasaketebeli
# end

# dur = time() - t
# rates = zeros(K)
# for i in 1:K
#     rates[i] = mean(distances .>= thresholds[i])
# end
# histogram(thresholds)