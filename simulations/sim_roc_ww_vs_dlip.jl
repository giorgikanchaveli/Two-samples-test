include("methods_fp_tp.jl")

using DelimitedFiles


function savefig_roc_iid_ww_dlip(n, m, save = false)
    @assert m == 1 "m must be equal to 1"
    # Problem parameters for iid
    p_a, p_b = 2.5, 4.0
    q_a, q_b = 2.5, 3.0
    p = Betapm(0.0,1.0,p_a,p_b)
    q = Betapm(0.0,1.0,q_a,q_b)

    # p = samepm(-1.0,1.0)
    # q = splittingpm(-1.0,1.0)

    n_rv = n
    fp_problem_pm = param_problem_pm(p, p, n_rv) # false positive problem for iid
    tp_problem_pm = param_problem_pm(p, q, n_rv) # true positive problem for iid

    # Problem parameters for exchangeable

    # p_1 = ()->probability("same")
    # p_2 = ()->probability("splitting")
    # dp_1 = DP(1.0, p_1, -1.0, 1.0)
    # dp_2 = DP(1.0, p_2, -1.0, 1.0)

    beta_p, beta_q = Beta(p_a, p_b), Beta(q_a, q_b)
    p_1 = ()->rand(beta_p)
    p_2 = ()->rand(beta_q)
    dp_1 = DP(1.0, p_1, -1.0, 1.0)
    dp_2 = DP(1.0, p_2, -1.0, 1.0)
    n_top, n_bottom = n_rv, m

    fp_problem_ppm = param_problem_ppm(dp_1, dp_1, n_top, n_bottom) # false positive problem for exchangeable
    tp_problem_ppm = param_problem_ppm(dp_1, dp_2, n_top, n_bottom) # true positive problem for exchangeable



    # simulation and algorithm parameter
    s = 50
    par_perm = param_perm(50)
    iter = s

    seed = 4567

    # compute rates

    println("computing rates for iid")
    α_iid = rej_rate(wass, fp_problem_pm, par_perm, iter, seed) # iid
    println("computation of α done")
    β_iid = rej_rate(wass, tp_problem_pm, par_perm, iter, seed) # iid
    println("computation of β done")

    println("computing rates for dlip")
    α_exch_dlip = rej_rate(dlip, fp_problem_ppm, par_perm, iter, seed) # exchangeable dlip
    println("computation of α done")
    β_exch_dlip = rej_rate(dlip, tp_problem_ppm, par_perm, iter, seed) # exchangeable dlip
    println("computation of β done")

    println("computing rates for ww")
    α_exch_ww = rej_rate(ww, fp_problem_ppm, par_perm, iter, seed) # exchangeable ww
    println("computation of α done")
    β_exch_ww = rej_rate(ww, tp_problem_ppm, par_perm, iter, seed) # exchangeable ww
    println("computation of β done")


    θ = collect(0.0:0.01:1.0)
    fp_plot = plot()
    plot!(fp_plot, θ, α_iid, label = "w",ratio = 1, xlims = (0,1), ylims = (0,1),
            xlabel="probability", ylabel="rejection rate", title="False Positive Rate, n = $n, m = $m")
    plot!(fp_plot, θ, α_exch_dlip, label = "dlip", ratio = 1, xlims = (0,1), ylims = (0,1),
    xlabel="probability", ylabel="rejection rate", title="False Positive Rate, n = $n, m = $m")
    plot!(fp_plot, θ, α_exch_ww, label = "ww", ratio = 1, xlims = (0,1), ylims = (0,1),
    xlabel="probability", ylabel="rejection rate", title="False Positive Rate, n = $n, m = $m")

    tp_plot = plot()
    plot!(tp_plot, θ, β_iid, label = "w",ratio = 1, xlims = (0,1), ylims = (0,1),
            xlabel="probability", ylabel="rejection rate", title="True Positive Rate, n = $n, m = $m")
    plot!(tp_plot, θ, β_exch_dlip, label = "dlip", ratio = 1, xlims = (0,1), ylims = (0,1),
    xlabel="probability", ylabel="rejection rate", title="True Positive Rate, n = $n, m = $m")
    plot!(tp_plot, θ, β_exch_ww, label = "ww", ratio = 1, xlims = (0,1), ylims = (0,1),
    xlabel="probability", ylabel="rejection rate", title="True Positive Rate, n = $n, m = $m")


    roc_plot = roc([α_iid, α_exch_dlip, α_exch_ww], [β_iid, β_exch_dlip, β_exch_ww],
                    ["w","dlip","ww"], "ROC curve, n = $n, m = $m")
    path = joinpath(pwd(), "both/plots")
    fp_path = joinpath(path, "n = $n, m = $m/fp_n=$n, m=$m.png")
    tp_path = joinpath(path, "n = $n, m = $m/tp_n=$n, m=$m.png")
    roc_path = joinpath(path, "n = $n, m = $m/roc_n=$n, m=$m.png")
    if save == true
        savefig(fp_plot, fp_path)
        savefig(tp_plot, tp_path)
        savefig(roc_plot, roc_path)
    end
    return ([α_iid, α_exch_dlip, α_exch_ww], [β_iid, β_exch_dlip, β_exch_ww], ["iid","dlip","ww"])
end




function savefig_roc_ww_dlip(n, m, save = false)

   
  
    # Problem parameters for exchangeable

    # p_1 = ()->probability("same")
    # p_2 = ()->probability("splitting")
    # dp_1 = DP(1.0, p_1, -1.0, 1.0)
    # dp_2 = DP(1.0, p_2, -1.0, 1.0)
    p_a, p_b, q_a, q_b = 1.2, 1.2, 1.1, 1.25
   # p_a, p_b = 1.2, 0.3
   # q_a, q_b = 1.1, 3.4
    # p_a, p_b = 2.5, 4.0
    # q_a, q_b = 2.5, 3.0
    beta_p, beta_q = Beta(p_a, p_b), Beta(q_a, q_b)
    p_1 = ()->rand(beta_p)
    p_2 = ()->rand(beta_q)
    dp_1 = DP(1.0, p_1, -1.0, 1.0)
    dp_2 = DP(1.0, p_2, -1.0, 1.0)
    n_top, n_bottom = n, m

    fp_problem_ppm = param_problem_ppm(dp_1, dp_1, n_top, n_bottom) # false positive problem for exchangeable
    tp_problem_ppm = param_problem_ppm(dp_1, dp_2, n_top, n_bottom) # true positive problem for exchangeable



    # simulation and algorithm parameter
    s = 100
    par_perm = param_perm(100)
    iter = s

    seed = 4567

    # compute rates
    println("computing rates for dlip")
    α_exch_dlip = rej_rate(dlip, fp_problem_ppm, par_perm, iter, seed) # exchangeable dlip
    println("computation of α done")
    β_exch_dlip = rej_rate(dlip, tp_problem_ppm, par_perm, iter, seed) # exchangeable dlip
    println("computation of β done")


    println("computing rates for ww")
    α_exch_ww = rej_rate(ww, fp_problem_ppm, par_perm, iter, seed) # exchangeable ww
    println("computation of α done")
    β_exch_ww = rej_rate(ww, tp_problem_ppm, par_perm, iter, seed) # exchangeable ww
    println("computation of β done")

    
    θ = collect(0.0:0.01:1.0)

    fp_plot = plot()
    plot!(fp_plot, θ, α_exch_dlip, label = "dlip", ratio = 1, xlims = (0,1), ylims = (0,1),
    xlabel="probability", ylabel="rejection rate", title="False Positive Rate, n = $n, m = $m")
    plot!(fp_plot, θ, α_exch_ww, label = "ww", ratio = 1, xlims = (0,1), ylims = (0,1),
    xlabel="probability", ylabel="rejection rate", title="False Positive Rate, n = $n, m = $m")

    tp_plot = plot()
    plot!(tp_plot, θ, β_exch_dlip, label = "dlip", ratio = 1, xlims = (0,1), ylims = (0,1),
    xlabel="probability", ylabel="rejection rate", title="True Positive Rate, n = $n, m = $m")
    plot!(tp_plot, θ, β_exch_ww, label = "ww", ratio = 1, xlims = (0,1), ylims = (0,1),
    xlabel="probability", ylabel="rejection rate", title="True Positive Rate, n = $n, m = $m")


    roc_plot = roc([α_exch_dlip, α_exch_ww], [β_exch_dlip, β_exch_ww],
                    ["dlip","ww"], "ROC curve, n = $n, m = $m")
    path = joinpath(pwd(), "both/plots")
    fp_path = joinpath(path, "n = $n, m = $m/fp_n=$n, m=$m.png")
    tp_path = joinpath(path, "n = $n, m = $m/tp_n=$n, m=$m.png")
    roc_path = joinpath(path, "n = $n, m = $m/roc_n=$n, m=$m.png")
    if save == true
        savefig(fp_plot, fp_path)
        savefig(tp_plot, tp_path)
        savefig(roc_plot, roc_path)
        # return raw data for latex
        fp_path_txt_dlip = joinpath(path, "n = $n, m = $m/fp_n=$n, m=$m, dlip.txt")
        fp_path_txt_ww = joinpath(path, "n = $n, m = $m/fp_n=$n, m=$m, ww.txt")
        # Combine data and headers
        data_dlip = [θ α_exch_dlip β_exch_dlip]
        data_ww = [θ α_exch_ww β_exch_ww]
        headers = ["thetas", "alphas", "betas"]

        # Save to the specified directory
        open(fp_path_txt_dlip, "w") do io
            write(io, join(headers, "\t") * "\n")  # Write headers
            writedlm(io, data_dlip, '\t')              # Write data
        end
        open(fp_path_txt_ww, "w") do io
            write(io, join(headers, "\t") * "\n")  # Write headers
            writedlm(io, data_ww, '\t')              # Write data
        end
        end
    return ([α_exch_dlip, α_exch_ww], [β_exch_dlip, β_exch_ww], ["dlip","ww"])
end

function create_roc_plot(αs, βs, plot_names)
    roc_plot = roc(αs, βs, plot_names, "ROC curve")
    return roc_plot
end

function create_rej_rate_plot(r::Vector{Vector{Float64}}, plot_names, title)
    θ = collect(0.0:0.01:1.0)
    pl = plot()
    for i in 1:length(r)
        plot!(pl, θ, r[i], label = plot_names[i], ratio = 1, xlims = (0,1), ylims = (0,1),
        xlabel="probability", ylabel="rejection rate", title=title)
    end
    return pl
end




t = time()
save = false
#savefig_roc_iid_ww_dlip(10, 1,save)
#savefig_roc_iid_ww_dlip(30, 1, save)
#savefig_roc_iid_ww_dlip(100, 1, save)

#savefig_roc_ww_dlip(10, 2, save)
α, β, str = savefig_roc_ww_dlip(256, 2, save)
α_dlip, α_ww = α[1], α[2]
β_dlip, β_ww = β[1], β[2]
#savefig_roc_ww_dlip(150, 2, save)
timeelapsed = time() - t
println("Time elapsed: $timeelapsed seconds")


#savefig_roc_ww_dlip(50, 5, save)
#savefig_roc_ww_dlip(150, 5, save)



roc_pl = create_roc_plot(α,β,str)
fp_pl = create_rej_rate_plot(α, str, "False Positive Rate")
tp_pl = create_rej_rate_plot(β, str, "True Positive Rate")  