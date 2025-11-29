include("methods.jl")
using Plots




println("running file tp_hipm_wow.jl")
println("number of threads: $(Threads.nthreads())")
println("expected duration is 3 hours")


# for S = 400, for one pair of RPMS total time should be 280 seconds. ( it is 200 seconds actually) (n = 100, m = 200, n_samples = 100, S = 400)
# for S = 400, for one pair of RPMS total time should be 280 seconds. ( it is 200 seconds actually) (n = 100, m = 200, n_samples = 100, S = 400)





bs = collect(1.05:0.05:1.5)
pairs = [(DP(1.0, Beta(1,1),0.0,1.0), DP(1.0, Beta(1,b),0.0,1.0)) for b in bs]
file_path = "plots/hipm_vs_wow"
title = "True Positive Rates"
xlabel = "b"
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
file_name = "tp_dpvaryingbetab_n=$(n)_m=$(m)_S=$(S)_$(resampling_method)_n_samples=$(n_samples)"
t = time()
save_fig_hipm_wow(pairs, bs, file_name, file_path, title, xlabel,ylabel, n,m,S,θ,n_samples,bootstrap)
#dur = time() - t
#println("total duration is $(dur/3600) hours")










αs = collect(1.1:5.0:100)
pairs = [(DP(1.0, Beta(1,1), 0.0, 1.0), DP(α, Beta(1,1), 0.0, 1.0)) for α in αs]
file_path = "plots/hipm_vs_wow"
title = "True Positive Rates"
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
file_name = "tp_DPvaryingalpha_n=$(n)_m=$(m)_S=$(S)_$(resampling_method)_n_samples=$(n_samples)"
save_fig_hipm_wow(pairs, αs, file_name, file_path, title, xlabel,ylabel, n,m,S,θ,n_samples,bootstrap)
dur = time() - t
println("total duration is $(dur/3600) hours")







