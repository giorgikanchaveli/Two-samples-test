include("methods.jl")


# define q_1, q_2
# function for plot rej rate. This one function includes fp,tp,roc because we will input title and two vectors.
# e.g. fp is θs and fps, roc is fps and tps. you also input x axis y axis. 


rej_rates_per_theta = function(q_1::PPM, q_2::PPM, n::Int, 
            m::Int, S::Int, n_samples::Int, bootstrap::Bool)
    θs = collect(0.0:0.01:1.0)
    rej_rates_hipm = zeros(length(θs))
    rej_rates_wow = zeros(length(θs))
    for s in 1:S
        h_1, h_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        rej_rates_hipm .+= decide_hipm(h_1, h_2, θs, n_samples, bootstrap)
        rej_rates_wow .+= decide_wow(h_1, h_2, θs, n_samples, bootstrap)
    end
    return rej_rates_hipm./S, rej_rates_hipm./S
end

q_1 = DP(1.0, Beta(1,1))
q_2 = DP(1.0, Beta(1,1.5))
n = 10
m = 10
S = 1
n_samples = 1
bootstrap = false

fp_hipm = rej_rates_per_theta_hipm(q_1, q_1, n, m, S, n_samples, bootstrap)
tp_hipm = rej_rates_per_theta_hipm(q_1, q_2, n, m, S, n_samples, bootstrap)

fp_wow = rej_rates_per_theta_hipm(q_1, q_1, n, m, S, n_samples, bootstrap)
tp_wow = rej_rates_per_theta_hipm(q_1, q_2, n, m, S, n_samples, bootstrap)


plot_rej_rate = function(rej_rates_hipm::Vector{Float64}, rej_rates_wow::Vector{Float64},
        title::String, xlabel::String, ylabel::String)
    θs = collect(0:0.01:1.0)
    pl = plot(title = title, ratio = 1.0, xlims = (0,1), ylims = (0,1),
         xlabel = xlabel, ylabel = ylabel)
    plot!(pl, [0, 1], [0, 1], linestyle=:dash, color=:gray,label=false)
    plot!(pl, θs, rej_rates_hipm, label = "HIPM")
    plot!(pl, θs, rej_rates_wow, label = "WoW")
    return pl
end
plot_roc = function(fp_hipm::Vector{Float64}, tp_hipm::Vector{Float64},
        fp_wow::Vector{Float64}, tp_wow::Vector{Float64},
        title::String, xlabel::String, ylabel::String)
    
    pl = plot(title = title, ratio = 1.0, xlims = (0,1), ylims = (0,1),
         xlabel = xlabel, ylabel = ylabel)
    plot!(pl, [0, 1], [0, 1], linestyle=:dash, color=:gray,label=false)
    plot!(pl, fp_hipm, tp_hipm, label = "HIPM")
    plot!(pl, fp_wow, tp_wow, label = "WoW")
    return pl
end

fp = plot_rej_rate(fp_hipm, fp_wow, "Type I error", "Significance level", "rate")
tp = plot_rej_rate(tp_hipm, tp_wow, "Power", "Significance level", "rate")
roc = plot_roc(fp_hipm, tp_hipm, fp_wow, tp_wow, "ROC", "Type I error", "Power")
savefig(fp, "plots/ch5/fp_n=$(n)_m=$(m)_S=$(S)_n_samples=$(n_samples).png")
savefig(tp, "plots/ch5/tp_n=$(n)_m=$(m)_S=$(S)_n_samples=$(n_samples).png")
savefig(roc, "plots/ch5/roc_n=$(n)_m=$(m)_S=$(S)_n_samples=$(n_samples).png")