include("methods.jl")
using Plots

# define q_1, q_2
# function for plot rej rate. This one function includes fp,tp,roc because we will input title and two vectors.
# e.g. fp is θs and fps, roc is fps and tps. you also input x axis y axis. 

if isempty(ARGS)
    @info "No arguments found. Running with local defaults."
    const n = 25
    const m = 100
    const b = 1.1
elseif length(ARGS) < 3
    error("Usage: julia fp_tp_roc.jl <n> <m> <b>")
else
    # These are picked up from the shell command line
    const n = parse(Int, ARGS[1])
    const m = parse(Int, ARGS[2])
    const b = parse(Float64, ARGS[3])
end



const n_sims = 1000
const n_samples = 100
const bootstrap = false




compute_rej_rates = function(q_1::LawRPM, q_2::LawRPM, n::Int, 
            m::Int, n_sims::Int, n_samples::Int, bootstrap::Bool)
    θs = collect(0.0:0.01:1.0)
    rej_rates_hipm = zeros(length(θs))
    rej_rates_wow = zeros(length(θs))

    @floop ThreadedEx() for s in 1:n_sims
        hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        @reduce rej_rates_hipm .+= decide_hipm(hier_sample_1, hier_sample_2, θs, n_samples, bootstrap)
        @reduce rej_rates_wow .+= decide_wow(hier_sample_1, hier_sample_2, θs, n_samples, bootstrap)
    end

    return rej_rates_hipm./n_sims, rej_rates_wow./n_sims
end


plot_rej_rate = function(rej_rates_hipm::Vector{Float64}, rej_rates_wow::Vector{Float64},
        title::String, xlabel::String, ylabel::String)
    θs = collect(0.0:0.01:1.0)
    pl = plot(title = title, aspect_ratio=:equal, xlims = (0,1), ylims = (0,1),
         xlabel = xlabel, ylabel = ylabel)
    plot!(pl, [0, 1], [0, 1], linestyle=:dash, color=:gray,label=false)
    plot!(pl, θs, rej_rates_hipm, label = "HIPM")
    plot!(pl, θs, rej_rates_wow, label = "WoW")
    return pl
end


plot_roc = function(fp_hipm::Vector{Float64}, tp_hipm::Vector{Float64},
        fp_wow::Vector{Float64}, tp_wow::Vector{Float64},
        title::String, xlabel::String, ylabel::String)
    
    pl = plot(title = title, aspect_ratio=:equal, xlims = (0,1), ylims = (0,1),
         xlabel = xlabel, ylabel = ylabel)
    plot!(pl, [0, 1], [0, 1], linestyle=:dash, color=:gray,label=false)
    plot!(pl, fp_hipm, tp_hipm, label = "HIPM")
    plot!(pl, fp_wow, tp_wow, label = "WoW")
    return pl
end


function run_experiment_and_save(n::Int, m::Int, b::Float64)

    q_1 = DP(1.0, Beta(1, 1), 0.0, 1.0)
    q_2 = DP(1.0, Beta(1, b), 0.0, 1.0)

    @info "Computing Rejection Rates..." n m b
    fp_hipm, fp_wow  = compute_rej_rates(q_1, q_1, n, m, n_sims, n_samples, bootstrap)
    tp_hipm, tp_wow = compute_rej_rates(q_1, q_2, n, m, n_sims, n_samples, bootstrap)


    fp_plot = plot_rej_rate(fp_hipm, fp_wow, "Type I error", "Significance level", "Type I error")
    tp_plot = plot_rej_rate(tp_hipm, tp_wow, "Power", "Significance level", "Power")
    roc_plot = plot_roc(fp_hipm, tp_hipm, fp_wow, tp_wow, "ROC", "Type I error", "Power")
    
    output_path = "plots/ch5"
    mkpath(output_path)

    # Identify the file by its parameters
    file_id = "n=$(n)_m=$(m)_b=$(b)"

    savefig(fp_plot,  joinpath(output_path, "fp_$(file_id).png"))
    savefig(tp_plot,  joinpath(output_path, "tp_$(file_id).png"))
    savefig(roc_plot, joinpath(output_path, "roc_$(file_id).png"))
    @info "Plots saved to $output_path"
end


t_start = time()
run_experiment_and_save(n, m, b)
@info "Total duration: $(round(time() - t_start, digits=2)) seconds"


