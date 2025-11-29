using Plots

include("methods.jl")



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

# total S = 400 for one pair is 960 seconds
        
# fig 1 with varying mean
println("running file fig_1_mean.jl")
println("number of threads: $(Threads.nthreads())")
println("expected duration is 7 hours")


δs = collect(-1.0:0.05:1.0)
pairs = [(tnormal_normal(0.0,0.5,-10.0,10.0), tnormal_normal(δ, 0.5, -10.0,10.0)) for δ in δs]
file_path = "plots/frechet/figure1"
title = "Rejection rates for 4 schemes"
xlabel = "δ"
ylabel = "Rej rate"
n = 100
m = 200
S = 500
n_samples = 200 # bootstrap\perm samples
θ = 0.05
bootstrap = false
file_name = "varying_mean_n=$(n)_m=$(m)_S=$(S)_permutation_n_samples=$(n_samples)"
t = time()
save_fig(pairs, δs, file_name, file_path, title, xlabel,ylabel, n,m,S,θ,n_samples,bootstrap)
dur = time() - t
println("total duration is $(dur/3600) hours")

