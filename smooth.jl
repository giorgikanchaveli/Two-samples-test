include("methods.jl")
using Plots




function save_fig(pairs::Vector{<:Tuple{PPM,PPM}}, param_pairs::Vector{Float64}, file_name::String, file_path::String, title::String, xlabel::String, ylabel::String,
    n::Int, m::Int, S::Int, θ::Float64, n_samples::Int, bootstrap::Bool)

    rates_wow = zeros(length(param_pairs))
    
    for i in 1:length(pairs)
        q_1, q_2 = pairs[i]
        rates_wow[i] = rejection_rate_wow(q_1,q_2,n,m,S,θ,n_samples,bootstrap)
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





τs = collect(0.1:0.05:3.0)
pairs = [(tnormal_normal(0.0,0.2,-10.0,10.0), tnormal_normal(0.0,0.2*τ,-10.0,10.0)) for τ in τs]
file_path = "plots/frechet/figure1"
title = "Rejection rates for wow"
xlabel = "τ"
ylabel = "Rej rate"
n = 100
m = 2000
S = 1000
n_samples = 1000
θ = 0.05
bootstrap = false
file_name = "varying_variance_n=$(n)_m=$(m)_S=$(S)_permutation_n_samples=$(n_samples)"
t = time()
save_fig(pairs, τs, file_name, file_path, title, xlabel,ylabel, n,m,S,θ,n_samples,bootstrap)
dur = time() - t
println("total duration is $(dur/3600) hours")
println("parameters are n = $(n), m = $(m), S = $(S), n_samples = $(n_samples), permutation")