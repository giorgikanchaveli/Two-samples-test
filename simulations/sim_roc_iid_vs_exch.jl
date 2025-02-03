include("methods_fp_tp.jl")







# Problem parameters for iid
# p_a, p_b = 2.5, 2.9
# q_a, q_b = 3.5, 4.2
# p = Betapm(0.0,1.0,p_a,p_b)
# q = Betapm(0.0,1.0,q_a,q_b)
p = samepm(-1.0,1.0)
q = splittingpm(-1.0,1.0)
n_rv = 30
fp_problem_pm = param_problem_pm(p, p, n_rv) # false positive problem for iid
tp_problem_pm = param_problem_pm(p, q, n_rv) # true positive problem for iid

# Problem parameters for exchangeable
p_1 = ()->probability("same")
p_2 = ()->probability("splitting")
dp_1 = DP(1.0, p_1, -1.0, 1.0)
dp_2 = DP(1.0, p_2, -1.0, 1.0)

# beta_p, beta_q = Beta(p_a, p_b), Beta(q_a, q_b)
# p_1 = ()->rand(beta_p)
# p_2 = ()->rand(beta_q)
# dp_1 = DP(1.0, p_1, -1.0, 1.0)
# dp_2 = DP(1.0, p_2, -1.0, 1.0)
n_top, n_bottom = n_rv, 1

fp_problem_ppm = param_problem_ppm(dp_1, dp_1, n_top, n_bottom) # false positive problem for exchangeable
tp_problem_ppm = param_problem_ppm(dp_1, dp_2, n_top, n_bottom) # true positive problem for exchangeable


# d_iid = sample_distances(wass, p, q, n_rv, 10, 12)

# d_exch = sample_distances(dlip, dp_1, dp_2, n_top, n_bottom, 10, 12)

# pl = plot()
# scatter!(pl, d_iid, label = "iid")
# scatter!(pl, d_exch, label = "exch")

# Random.seed!(123)
# p_emp_iid, q_emp_iid = generate_emp(p, n_rv), generate_emp(q, n_rv)
# Random.seed!(123)
# p_exch_iid, q_exch_iid = generate_emp(dp_1, n_top, n_bottom), generate_emp(dp_2, n_top, n_bottom)
# perm = param_perm(10)
# θs = collect(0.0:0.01:1.0)
# thresh_iid = perm_thresholds(wass, p_emp_iid, q_emp_iid, perm, θs, 12)
# thresh_exch = perm_thresholds(dlip, p_exch_iid, q_exch_iid, perm, θs, 12)

# pl = plot()
# scatter!(pl, thresh_iid, label = "iid")
# scatter!(pl, thresh_exch, label = "exch")


# parameters for permutation approach



s = 100
par_perm = param_perm(100)

# iid_thresholds = perm_thresholds(wass, p_emp_iid, q_emp_iid, par_perm, collect(0.0:0.01:1.0),12)

# exch_thresholds = perm_thresholds(dlip, p_exch_iid, q_exch_iid, par_perm, collect(0.0:0.01:1.0),12)

# r_iid = [mean(sqrt(n_rv/2)*d_iid .> t) for t in iid_thresholds]
# r_exch = [mean(sqrt(n_rv/2)*d_exch .> t) for t in exch_thresholds]

# pl = plot()
# plot!(pl, collect(0.0:0.01:1.0), r_iid, label = "iid")
# plot!(pl, collect(0.0:0.01:1.0), r_exch, label = "exch")

# simulation parameters
θ = collect(0.0:0.01:1.0)
iter = s





# # ise


# seed = 467168
# α_iid_fake = rej_rate(wass, fp_problem_pm, par_perm, iter, seed)
# β_iid_fake = rej_rate(wass, tp_problem_pm, par_perm, iter, seed)

# seed = 467169
# α_iid_exact = rej_rate_exact(wass, fp_problem_pm, par_perm, iter, seed)
# β_iid_exact = rej_rate_exact(wass, tp_problem_pm, par_perm, iter, seed)

# roc_plot_iid = roc([α_iid_fake, α_iid_exact], [β_iid_fake, β_iid_exact], "fake","exact")
# # ise


# rejection rate for the permutation approach under null and alternative hypothesis
seed = 4567

α_iid = rej_rate(wass, fp_problem_pm, par_perm, iter, seed)
β_iid = rej_rate(wass, tp_problem_pm, par_perm, iter, seed)

α_exch = rej_rate(dlip, fp_problem_ppm, par_perm, iter, seed)
β_exch = rej_rate(dlip, tp_problem_ppm, par_perm, iter, seed)

pl_α = plot()
plot!(pl_α, θ, α_iid, label = "iid")
plot!(pl_α, θ, α_exch, label = "exch")

pl_β = plot()
plot!(pl_β, θ, β_iid, label = "iid")
plot!(pl_β, θ, β_exch, label = "exch")



roc_plot_iid = roc([α_iid, α_exch], [β_iid, β_exch], "iid","exch")
display(roc_plot_iid)




# fp_plot_iid = plot(θ, α_iid, ratio = 1, xlims = (0,1), ylims = (0,1), xlabel="probability", ylabel="rejection rate", title="False Positive Rate,IID", 
#                 label = "perm")
# tp_plot_iid = plot(θ, β_iid, ratio = 1, xlims = (0,1), ylims = (0,1), xlabel="probability", ylabel="rejection rate", title="True Positive Rate,IID", 
#                 label = "perm")
# roc_plot_iid = plot(α_iid, β_iid, ratio = 1, xlims = (0,1), ylims = (0,1), xlabel="false positive rate", ylabel="true positive rate", title="ROC curve,IID", 
#                 label = "perm")

# fp_plot_exch = plot(θ, α_exch, ratio = 1, xlims = (0,1), ylims = (0,1), xlabel="probability", ylabel="rejection rate", title="False Positive Rate,Exch", 
#                 label = "perm")
# tp_plot_exch = plot(θ, β_exch, ratio = 1, xlims = (0,1), ylims = (0,1), xlabel="probability", ylabel="rejection rate", title="True Positive Rate,Exch", 
#                 label = "perm")
# roc_plot_exch = plot(α_exch, β_exch, ratio = 1, xlims = (0,1), ylims = (0,1), xlabel="false positive rate", ylabel="true positive rate", title="ROC curve,Exch", 
#                 label = "perm")









# # Problem parameters
# prob_0 = ()->probability("same")
# prob_1 = ()->probability("splitting")
# b_1 = Beta(2.5,2.9)
# b_2 = Beta(3.5, 4.2)
# b_aux = Beta(8.5,1.2)

# Random.seed!(1234)

# un_1 = ()->rand(b_1)
# un_2 = ()->rand(b_2)
# un_aux = ()->rand(b_aux)

# p = DP(1.0, un_1, -1.0, 1.0)
# q = DP(1.0, un_2, -1.0, 1.0)

# n_top, n_bottom = 300, 1
# # n = n_top
# # m = n_bottom

# fp_problem = param_problem_ppm(p, p, n_top, n_bottom) # false positive problem
# tp_problem = param_problem_ppm(p, q, n_top, n_bottom) # true positive problem
# # dist = dlip


# # parameters for the empirical and permutation approaches
# p_aux = DP(3.0, prob_0, -1.0, 1.0)
# s = 170

# par_emp = param_emp_ppm(p_aux, s)

# # test_stats_emp = sqrt(n / 2) * sample_distances(dist, p_aux, p_aux, n_top, n_bottom, iter)
# # thresholds_emp = quantile(test_stats_emp, 1 .- θ)

# # p_emp,q_emp = generate_emp(p, n_top, n_bottom), generate_emp(q, n_top, n_bottom)
# # x,y = vec(p_emp.atoms), vec(q_emp.atoms)
# # dww = ww(p_emp, q_emp)
# # dhh = dlip(p_emp, q_emp)
# # d = mean(abs.(sort(x)-sort(y)))

# par_perm = param_perm(100)

# # simulation parameters
# θ = collect(0.0:0.01:1.0)
# iter = s

# # rejection rates for the empirical and permutation approaches under null and alternative hypothesis
# t = time()


# # test_stats_fp = sqrt(n / 2) * sample_distances(dist, p, p, n, m, iter)
# # test_stats_tp = sqrt(n / 2) * sample_distances(dist, p, q, n, m, iter)

# # α_emp = rej_rate(dlip, test_stats_fp, thresholds_emp, fp_problem, par_emp, iter)
# # β_emp = rej_rate(dlip, test_stats_tp, thresholds_emp, tp_problem, par_emp, iter)
# # α_perm = rej_rate(dlip, test_stats_fp, fp_problem, par_perm, iter)
# # β_perm = rej_rate(dlip, test_stats_fp, tp_problem, par_perm, iter)




# #α_emp = rej_rate(dlip, fp_problem, par_emp, iter)
# #β_emp = rej_rate(dlip, tp_problem, par_emp, iter)
# α_perm_dlip = rej_rate(dlip, fp_problem, par_perm, s, 1234)
# β_perm_dlip = rej_rate(dlip, tp_problem, par_perm, s, 1234)
# α_perm_dww = rej_rate(ww, fp_problem, par_perm, s, 1234)
# β_perm_dww = rej_rate(ww, tp_problem, par_perm, s,1234)

# t = time() - t

# #roc_plot = roc([α_emp, α_perm], [β_emp, β_perm])
# #display(roc_plot)

# # fp_emp_plot = plot(θ, α_emp, ratio = 1, xlims = (0,1), ylims = (0,1), xlabel="probability", ylabel="rejection rate", title="False Positive Rate", 
# #                 label = "emp")
# # tp_emp_plot = plot(θ, β_emp, ratio = 1, xlims = (0,1), ylims = (0,1), xlabel="probability", ylabel="rejection rate", title="True Positive Rate", 
# #               label = "emp")
# # roc_emp_plot = plot(α_emp, ratio = 1, xlims = (0,1), ylims = (0,1), β_emp, xlabel="false positive rate", ylabel="true positive rate", title="ROC curve", 
# #               label = "emp")
# fp_dlip_perm_plot = plot(θ, α_perm_dlip, ratio = 1, xlims = (0,1), ylims = (0,1), xlabel="probability", ylabel="rejection rate", title="False Positive Rate, dLip", 
#                 label = "perm, n = $n_top, m = $n_bottom")  
# tp_dlip_perm_plot = plot(θ, β_perm_dlip, ratio = 1, xlims = (0,1), ylims = (0,1), xlabel="probability", ylabel="rejection rate", title="True Positive Rate, dLip", 
#                 label = "perm, n = $n_top, m = $n_bottom")
# roc_dlip_perm_plot = plot(α_perm_dlip, β_perm_dlip, ratio = 1, xlims = (0,1), ylims = (0,1), xlabel="false positive rate", ylabel="true positive rate", title="ROC curve, dLip", 
#                 label = "perm, n = $n_top, m = $n_bottom")

# fp_dww_perm_plot = plot(θ, α_perm_dww, ratio = 1, xlims = (0,1), ylims = (0,1), xlabel="probability", ylabel="rejection rate", title="False Positive Rate, WW", 
#                 label = "perm, n = $n_top, m = $n_bottom")  
# tp_dww_perm_plot = plot(θ, β_perm_dww, ratio = 1, xlims = (0,1), ylims = (0,1), xlabel="probability", ylabel="rejection rate", title="True Positive Rate, WW", 
#                 label = "perm, n = $n_top, m = $n_bottom")
# roc_dww_perm_plot = plot(α_perm_dww, β_perm_dlip, ratio = 1, xlims = (0,1), ylims = (0,1), xlabel="false positive rate", ylabel="true positive rate", title="ROC curve, WW", 
#                 label = "perm, n = $n_top, m = $n_bottom")

# roc_plot = roc([α_perm_dlip, α_perm_dww], [β_perm_dlip, β_perm_dww],"dlip","ww")
# #display(roc_plot)

# # current_dir = pwd()
# # file_path_fp = joinpath(current_dir, "plots/fp_perm_$(n_top)_$(n_bottom).png")
# # file_path_tp = joinpath(current_dir, "plots/tp_perm_$(n_top)_$(n_bottom).png")
# # file_path_roc = joinpath(current_dir, "plots/roc_perm_$(n_top)_$(n_bottom).png")
# # savefig(fp_perm_plot, file_path_fp)
# # savefig(tp_perm_plot, file_path_tp)
# # savefig(roc_perm_plot, file_path_roc)

# # Defining DP

# # function fptp_dir_proc(α::Float64, s::Int, n_top::Int, n_bottom::Int, par::param_perm)
# #     p_0 = ()->probability("same")
# #     p_1 = ()->probability("splitting")
# #     q_1 = DP(α, p_0, -1.0, 1.0)
# #     q_2 = DP(α, p_1, -1.0, 1.0)
# #     rej_rate = rejection_rate(q_1, q_1, q_1, q_2, s, n_top, n_bottom, par)
# #     return rej_rate
# # end



# # n_top = 50
# # n_bottom = 1
# # s = 150
# # perm = param_perm(110)


# # Random.seed!(1234)
# # t = time()
# # rej_rate = fptp_dir_proc(1.0, s, n_top, n_bottom, par)
# # t = time() - t
# # println("time: $t")

# # fp = rej_rate[1,:]
# # tp = rej_rate[2,:]

# # θs = collect(0.0:0.01:1.0)
# # p = plot()
# # plot!(p, θs, fp, xlabel="probability", ylabel="rejection rate", title="fp vs tp", label = "fp")
# # plot!(p, θs, tp, xlabel="probability", ylabel="rejection rate", title="fp vs tp", label = "tp")
# # display(p)





# # function plot_fp_tp(s::Int, n_top::Int, n_bottom::Int, params_perm::param_perm, k::Int)
# #     α = 1.0
# #     rpm_same_1 = DP(α, prob_1, -1.0, 1.0)
# #     rpm_same_2 = DP(α, prob_1, -1.0, 1.0)
# #     title = ["Permutation", "Threshold"]
# #     rpm_diff_1 = DP(α, prob_1, -1.0, 1.0)
# #     rpm_diff_2 = DP(α, prob_2, -1.0, 1.0)
# #     rej_rate = rejection_rate(rpm_same_1, rpm_same_2, rpm_diff_1, rpm_diff_2, s, 
# #         n_top, n_bottom, params_perm)
# #     # ROC curve for permutation and threshold

# #     roc_plot = plot(rej_rate[1,:,k], rej_rate[2,:,k], title = "ROC curve for $(title[k]), n = $n_top",legend = false,
# #             xlabel = "False positive rate", ylabel = "True positive rate")

# #     # FP rates for permutation and threshold
# #     θs = collect(0.0:0.01:1.0)
# #     fp_plot = plot()
# #     plot!(fp_plot, θs, rej_rate[1,:,k], title = "FP rates for $(title[k]) n = $n_top",legend = false,
# #             xlabel = "θ", ylabel = "FP rate")

# #     # TP rates for permutation and threshold
# #     tp_plot = plot()
# #     plot!(tp_plot, θs, rej_rate[2,:,k], title = "TP rates for $(title[k]) n = $n_top",legend = false,
# #             xlabel = "θ", ylabel = "TP rate")
# #     return([roc_plot, fp_plot, tp_plot], rej_rate)







# #     # roc_plot = plot(rej_rate[1,:,1], rej_rate[2,:,1], title = "ROC curve",label = "Permutation",
# #     #         xlabel = "False positive rate", ylabel = "True positive rate")
# #     # plot!(roc_plot, rej_rate[1,:,2], rej_rate[2,:,2],label = "Threshold")

# #     # # FP rates for permutation and threshold
# #     # θs = collect(0.0:0.01:1.0)
# #     # fp_plot = plot()
# #     # plot!(fp_plot, θs, rej_rate[1,:,1], title = "FP rates", label = "Permutation",
# #     #         xlabel = "θ", ylabel = "FP rate")
# #     # plot!(fp_plot, θs, rej_rate[1,:,2], label = "Threshold")

# #     # # TP rates for permutation and threshold
# #     # tp_plot = plot()
# #     # plot!(tp_plot, θs, rej_rate[2,:,1], title = "TP rates", label = "Permutation",
# #     #         xlabel = "θ", ylabel = "TP rate")
# #     # plot!(tp_plot, θs, rej_rate[2,:,2], label = "Threshold")
# #     # return([roc_plot, fp_plot, tp_plot], rej_rate)
# #     return p, rej_rate
# # end


# # # t = time()
# # # ns = [1, 5, 10]
# # # s = 50


# # function save_figures(ns::Vector{Int}, s::Int, k::Int)
# #     title = ["Permutation", "Threshold"]
# #     p_perm = params_perm(100)
# #     current_dir = pwd()
# #     for n in ns
# #         println("n = $n")
# #         l, rej_rate = plot_fp_tp(s, n, n, p_perm, k)
# #         roc_plot, fp_plot, tp_plot = l[1], l[2], l[3]

# #         folder_path = joinpath(current_dir, "test/$(title[k])")
# #         file_path_roc = joinpath(folder_path, "roc_$(title[k])_n = $n.png")
# #         file_path_fp = joinpath(folder_path, "fp_$(title[k])_n = $n.png")
# #         file_path_tp = joinpath(folder_path, "tp_$(title[k])_n = $n.png")

# #         savefig(roc_plot, file_path_roc)
# #         savefig(fp_plot, file_path_fp)
# #         savefig(tp_plot, file_path_tp)
# #     end
# # end

# # # save_figures(ns, s, 1)
# # # save_figures(ns, s, 2)

# # # t = time()-t
# # # print("Time taken: $t")



# # # s = 1 300 wami
