include("methods.jl")
using Plots





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
    fig = plot(title = title, xlabel = xlabel, ylabel = ylabel, xlims=(minimum(param_pairs) - 0.05, maximum(param_pairs)+ 0.05),
                         ylims = (-0.1, 1.1))
    plot!(fig, param_pairs, rates_hipm, label = "hipm", color = "green", marker = (:circle, 4))
    plot!(fig, param_pairs, rates_wow, label = "wow", color = "brown", marker = (:circle, 4))
    filepath = joinpath(pwd(), file_path)
    savefig(fig,joinpath(filepath, file_name))
end


println("running file fp_hipm_vs_wow.jl")
println("number of threads: $(Threads.nthreads())")
println("expected duration is 6 hours")


# for S = 400, for one pair of RPMS total time should be 280 seconds. ( it is 200 seconds actually) (n = 100, m = 200, n_samples = 100, S = 400)
# for S = 400, for one pair of RPMS total time should be 280 seconds. ( it is 200 seconds actually) (n = 100, m = 200, n_samples = 100, S = 400)


αs = collect(1.0:5.0:100)
pairs = [(DP(α, Beta(1,1), 0.0, 1.0), DP(α, Beta(1,1), 0.0, 1.0)) for α in αs]
file_path = "plots/hipm_vs_wow"
title = "False Positive Rates"
xlabel = "α"
ylabel = "Rej rate"
n = 100
m = 200
S = 400
n_samples = 200 # bootstrap\perm samples
θ = 0.05
bootstrap = false
if bootstrap
    resampling_method = "bootstrap"
else
    resampling_method = "permutation"
end
file_name = "fp_DPvaryingalpha_n=$(n)_m=$(m)_S=$(S)_$(resampling_method)_n_samples=$(n_samples)"
t = time()
save_fig_hipm_wow(pairs, αs, file_name, file_path, title, xlabel,ylabel, n,m,S,θ,n_samples,bootstrap)
#dur = time() - t
#println("total duration is $(dur/3600) hours")


τs = collect(1.0:0.25:3)
pairs = [(tnormal_normal(0.0, τ, -10.0, 10.0),tnormal_normal(0.0, τ, -10.0, 10.0)) for τ in τs]
file_path = "plots/hipm_vs_wow"
title = "False Positive Rates"
xlabel = "τ"
ylabel = "Rej rate"
n = 100
m = 200
S = 400
n_samples = 200 # bootstrap\perm samples
θ = 0.05
bootstrap = false
if bootstrap
    resampling_method = "bootstrap"
else
    resampling_method = "permutation"
end
file_name = "fp_tnormnormvariance_n=$(n)_m=$(m)_S=$(S)_$(resampling_method)_n_samples=$(n_samples)"
t = time()
save_fig_hipm_wow(pairs, τs, file_name, file_path, title, xlabel,ylabel, n,m,S,θ,n_samples,bootstrap)
dur = time() - t
println("total duration is $(dur/3600) hours")


