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



# for one pair, S = 400, n = 100, m = 200, n_samples = 200 : 

println("running file fig_counterexample.jl")
println("number of threads: $(Threads.nthreads())")
println("expected duration is 30 hours")



# counterexample 

λs = collect(0.0:0.05:1.0)
q_1 = simple_discr_1()
q_2_aux = simple_discr_2()
pairs = [(q_1, mixture_ppm(q_1, q_2_aux, λ)) for λ in λs]
file_path = "plotscluster"
title = "Rejection rates for 4 schemes"
xlabel = "λ"
ylabel = "Rej rate"
n = 100
m = 200
S = 1000
n_samples = 100
θ = 0.05
bootstrap = false
file_name = "counterexample_n=$(n)_m=$(m)_S=$(S)_permutation_n_samples=$(n_samples)"
println("parameters are S = $S, n_samples = $(n_samples), n = n, m = m, n_threads = $(Threads.nthreads())")
println("number of pairs of laws of RPMS: $(length(pairs))")
t = time()
save_fig(pairs, λs, file_name, file_path, title, xlabel,ylabel, n,m,S,θ,n_samples,bootstrap)
dur = time() - t
println("total duration is $(dur/3600) hours")
