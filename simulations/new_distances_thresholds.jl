include("../approaches/emp_threshold_approach.jl")
include("../approaches/permutation_approach.jl")
include("../distances/distance_Wasserstein.jl")
include("../distances/new_distance.jl")
include("../distances/w_distance.jl")
using QuadGK



function find_betas(d::Float64)
    # Given a value d, find two beta distributions with wass distance between them close to d
    # First beta distribution is fixed, for the second, only the parameter a is fixed

    # if found parameters, returns them, otherwise returns false
    # can find parameters for the distances in (0.0, 0.7)
    p_a, p_b = 1.0, 0.3
    a_range = collect(0.1:0.1:2.0)
    b_range = collect(0.1:0.1:2.0)
    ϵ = 0.01
    for a in a_range
        for b in b_range
            p = Beta(p_a, p_b)
            q = Beta(a, b)
            f = (x) -> abs(cdf(p, x) - cdf(q, x))
            d_w = quadgk(f, 0.0, 1.0)[1]
            if abs(d_w - d) < ϵ
                return p_a, p_b, a, b
            end
        end
    end
    return -1.0,-1.0,-1.0,-1.0
end


# methods for sampling
function direct_sampling(p::PPM, q::PPM, n_top::Int, n_bottom::Int, nReps::Int)
    # Given two laws of RPM we simulate the distances between empirical measures 
  

    d_ww, d_lip = zeros(nReps), zeros(nReps)
    for i in 1:nReps
        if i % 10 == 0
            println("iteration (direct): $i")
        end
        p_emp, q_emp = generate_emp(p, n_top, n_bottom), generate_emp(q, n_top, n_bottom)
        d_ww[i] = ww(p_emp, q_emp) # copy(p_emp) isn't necessary
        d_lip[i] = dlip(p_emp, q_emp) # copy(p_emp) isn't necessary
    end
    return d_ww, d_lip
end

function pooled_measure_sampling(p::PPM, q::PPM, n_top::Int, n_bottom::Int, nReps::Int)
    # Given two laws of RPM we simulate the distances between empirical measures 
    # from pooled measure (p+q)/2

    
    d_ww, d_lip = zeros(nReps), zeros(nReps)
    for i in 1:nReps
        if i % 10 == 0
            println("iteration (pooled): $i")
        end
        atoms_p, atoms_q = zeros(n_top, n_bottom), zeros(n_top, n_bottom)
        for j in 1:n_top
            if rand() < 0.5
                atoms_p[j,:] = dirichlet_process_without_weight(n_bottom, p.α, p.p_0)   
            else
                atoms_p[j,:] = dirichlet_process_without_weight(n_bottom, q.α, q.p_0)
            end
            if rand() < 0.5
                atoms_q[j,:] = dirichlet_process_without_weight(n_bottom, p.α, p.p_0)
            else
                atoms_q[j,:] = dirichlet_process_without_weight(n_bottom, q.α/2, q.p_0)
            end
        end

        p_emp = emp_ppm(atoms_p, n_top, n_bottom, p.a, p.b)
        q_emp = emp_ppm(atoms_q, n_top, n_bottom, q.a, q.b)
       
        d_ww[i] = ww(p_emp, q_emp)
        d_lip[i] = dlip(p_emp, q_emp)
    end
    return d_ww, d_lip
end




function plot_quantiles(d_ww, d_lip, title, labels)
    # plot quantiles of the distances from ww and lip
    θs = collect(0.0:0.01:1.0)
    q_plot = plot(title=title)
    plot!(q_plot, θs, quantile(d_ww, θs), xlabel="probability", ylabel="quantiles", label = labels[1])
    plot!(q_plot, θs, quantile(d_lip, θs), xlabel="probability", ylabel="quantiles", label = labels[2])
    return q_plot
end

function plot_vectors(v::Vector{Vector{Float64}}, title, labels)
    # plot the vectors
    θs = collect(0.0:0.01:1.0)
    p = plot(title = title, xlabel = "probability", ylabel = "values", ratio = 1.0,xlims = (0,1), ylims = (0,1))
    for i in 1:length(v)
        plot!(p, θs, v[i], label = labels[i])
    end
    return p
end



# Define measures
function savefigures(d, n_top, n_bottom)
    p_a, p_b, q_a, q_b = find_betas(d)
    p_1 = ()->rand(Beta(p_a, p_b))
    p_2 = ()->rand(Beta(q_a, q_b))
    α = 1.0
    dp_1 = DP(α, p_1, 0.0, 1.0)
    dp_2 = DP(α, p_2, 0.0, 1.0)

    # compute distances using direct samples
    nReps = 125 # 75
    d_ww, d_lip = direct_sampling(dp_1, dp_2, n_top, n_bottom, nReps)

    θs = collect(0.0:0.01:1.0)
    nPerms = 50 # 30
    t = time()
    d_ww_perm, d_lip_perm = pooled_measure_sampling(dp_1, dp_2, n_top, n_bottom, nPerms)
    t = time() - t
    thresh_ww = quantile(d_ww_perm, 1 .- θs)
    thresh_lip = quantile(d_lip_perm, 1 .- θs)

    thresh_plots = plot_vectors([thresh_ww, thresh_lip], "thresholds, samesplitting", ["ww", "lip"])
    d_ww_quantiles = quantile(d_ww, 1 .- θs)
    d_lip_quantiles = quantile(d_lip, 1 .- θs)

    rej_ww = [mean(d_ww .> t) for t in thresh_ww]
    rej_lip = [mean(d_lip .> t) for t in thresh_lip]

    thresh_rej_plots = plot_vectors([thresh_ww, thresh_lip, rej_ww, rej_lip],
                        "thresholds and rejection rates, samesplitting", ["ww threshold", "lip threshold", "ww rejection", "lip rejection"])


    thresh_quant_plots = plot_vectors([thresh_ww, thresh_lip, d_ww_quantiles, d_lip_quantiles],
                        "thresholds and quantiles, samesplitting", 
                        ["ww threshold", "lip threshold", "ww quantiles", "lip quantiles"])
    filepath = joinpath(pwd(), "simulations/axaliplotebi")
    savefig(thresh_quant_plots, joinpath(filepath, "d = $(d)/n = $(n_top), m = $(n_bottom)/thresh_quant_$(d)_$(n_top)_$(n_bottom).png"))
    savefig(thresh_rej_plots, joinpath(filepath, "d = $(d)/n = $(n_top), m = $(n_bottom)/thresh_rej_$(d)_$(n_top)_$(n_bottom).png"))
end

# savefigures(0.1,16,2)
# savefigures(0.1,128,2)
# savefigures(0.1,16,5000)
savefigures(0.1,128,5000)


savefigures(0.5,16,2)
savefigures(0.5,128,2)
savefigures(0.5,16,5000)
savefigures(0.5,128,5000)







# p_a, p_b, q_a, q_b = find_betas(0.3)
# p_1 = ()->rand(Beta(p_a, p_b))
# p_2 = ()->rand(Beta(q_a, q_b))
# α = 1.0
# dp_1 = DP(α, p_1, 0.0, 1.0)
# dp_2 = DP(α, p_2, 0.0, 1.0)

# n_top, n_bottom = 16, 2

# # compute distances using direct samples
# nReps = 75
# d_ww, d_lip = direct_sampling(dp_1, dp_2, n_top, n_bottom, nReps)



# θs = collect(0.0:0.01:1.0)
# nPerms = 30
# t = time()
# d_ww_perm, d_lip_perm = pooled_measure_sampling(dp_1, dp_2, n_top, n_bottom, nPerms)
# t = time() - t
# thresh_ww = quantile(d_ww_perm, 1 .- θs)
# thresh_lip = quantile(d_lip_perm, 1 .- θs)

# thresh_plots = plot_vectors([thresh_ww, thresh_lip], "thresholds, samesplitting", ["ww", "lip"])
# d_ww_quantiles = quantile(d_ww, 1 .- θs)
# d_lip_quantiles = quantile(d_lip, 1 .- θs)

# rej_ww = [mean(d_ww .> t) for t in thresh_ww]
# rej_lip = [mean(d_lip .> t) for t in thresh_lip]

# thresh_rej_plots = plot_vectors([thresh_ww, thresh_lip, rej_ww, rej_lip],
#                      "thresholds and rejection rates, samesplitting", ["ww threshold", "lip threshold", "ww rejection", "lip rejection"])






# thresh_quant_plots = plot_vectors([thresh_ww, thresh_lip, d_ww_quantiles, d_lip_quantiles],
#                      "thresholds and quantiles, samesplitting", 
#                      ["ww threshold", "lip threshold", "ww quantiles", "lip quantiles"])
# # using above functions generate the plots of intereset

