using Plots
include("approaches/emp_threshold_approach.jl")
include("approaches/permutation_approach.jl")
include("distance_Wasserstein.jl")


### 1) Distribution of the distances between the hierachical empirical 
#      measures with direct and permutation sampling


# Histograms

function hist_perm_vs_direct(dist::Function, p::PPM, q::PPM, n::Int, m::Int, s::Int)
    # this function plots the histograms of the distances between the empirical measures
    # for both direct sampling and permutation approach
    # It also ensures that x axis is the same for both histograms

    # s : number of samples for both direct sampling and permutation approach
    # n,m : n_top, n_bottom
    perm = param_perm(s)
    d_direct = sqrt(n/2)*sample_distances(dist, p, q, n, m, s)
    d_perm = sqrt(n/2)*sample_distances(dist, p, q, n, m, perm)

    x_max = maximum([maximum(d_direct), maximum(d_perm)]) # sets the limit of the x-axis 
    
    h_direct = histogram(d_direct, label="distances", xlabel="distance", ylabel="frequency",
                 title="Histogram of distances between empirical measures",
                 xticks=0.0:0.2:3.0, xlims = (0, x_max),bins = 30)
    vline!(h_direct, [mean(d_direct)], label="mean", color="red")

    h_perm = histogram(d_perm, label="distances", xlabel="distance", ylabel="frequency",
                 title="Histogram of distances between permuted empirical measures",
                 xticks=0.0:0.2:3.0, xlims = (0, x_max), bins = 30)
     
    vline!(h_perm, [mean(d_perm)], label="mean", color="red")

    return d_direct, d_perm, h_direct, h_perm
end

# Quantile Plots

function q_plot_direct_vs_perm(dist::Function, p::PPM, q::PPM, n::Int, m::Int, s::Int)
    # this function plots the quantile plots of the distances between the empirical measures
    # for both direct sampling and permutation approach

    # s : number of samples for both direct sampling and permutation approach
    # n,m : n_top, n_bottom
    perm = param_perm(s)
    d_direct = sqrt(n/2)*sample_distances(dist, p, q, n, m, s)
    d_perm = sqrt(n/2)*sample_distances(dist, p, q, n, m, perm)
    
    θs = collect(0.0:0.01:1.0)
    q_direct, q_perm = quantile(d_direct, θs), quantile(d_perm, θs)
    q_plot = plot(θs, q_direct, xlabel="probability", ylabel="quantiles", title="Q plot", label = "direct")
    plot!(q_plot, θs, q_perm, xlabel="probability", ylabel="quantiles", title="Q plot", label = "perm")
    
    return q_plot
end





p_0 = ()->probability("same")
p_1 = ()->probability("splitting")
p = DP(1.0, p_0, -1.0, 1.0)
q = DP(2.0, p_1, -1.0, 1.0)
n,m = 50,1

s = 5



# WW vs dlip
# d_l,d_ww = zeros(20), zeros(20)
# for i in 1:20    
#     p_emp, q_emp = generate_emp(p, n, m), generate_emp(p, n, m)
#     d_l[i] = dlip(p_emp, q_emp)
#     d_ww[i] = ww(p_emp, q_emp)
# end

# pl = plot()
# scatter!(pl, d_l, label = "dlip")
# scatter!(pl, d_ww, label = "ww")




# Rejection rates for WW and dlip

perm = param_perm(s)


# Test statistics
ts_dl = sqrt(n/2)*sample_distances(dlip, p, q, n, m, s)
ts_ww = sqrt(n/2)*sample_distances(ww, p, q, n, m, s)


pl = plot()
scatter!(pl, sort(ts_dl)./sqrt(n/2), label = "dlip")
scatter!(pl, sort(ts_ww)./sqrt(n/2),label = "ww")



# Get thresholds
θs = collect(0.0:0.01:1.0)
p_emp, q_emp = generate_emp(p, n, m), generate_emp(q, n, m)
Random.seed!(1234)
d_perms_lip = sqrt(n/2)*sample_distances(dlip, p_emp, q_emp, perm)
Random.seed!(1234)
d_perms_ww = sqrt(n/2)*sample_distances(ww, p_emp, q_emp, perm)

thresholds_lip = quantile(d_perms_lip, 1 .- θs)
thresholds_ww = quantile(d_perms_ww, 1 .- θs)

r_lip = [mean(ts_dl .> t) for t in thresholds_lip]
r_ww = [mean(ts_ww .> t) for t in thresholds_ww]

pl_r = plot(θs, r_lip, ratio = 1, xlims = (0,1), ylims = (0,1), xlabel="probability", ylabel="rejection rate", 
            title="Rejection rate vs probability", label = "dlip")
            
plot!(pl_r, θs, r_ww, ratio = 1, xlims = (0,1), ylims = (0,1), xlabel="probability", ylabel="rejection rate",
            title="Rejection rate vs probability", label = "ww")



# rates should be exactly same













perm = param_perm(s)

t = time()
#qplot = q_plot_direct_vs_perm(dlip, p, q, n, m, s)
d_direct, d_perm, h_direct, h_perm = hist_perm_vs_direct(dlip, p, p, n, m, s)

t = time() - t
print("time is ", t)

plot(h_direct, h_perm, layout=(2,1), link = :x)

θs = collect(0.0:0.01:1.0)
thresholds_perm = quantile(d_perm, 1 .- θs)
r = [mean(d_direct .> t) for t in thresholds_perm]

r_plot = plot(θs, r, ratio = 1, xlims = (0,1), ylims = (0,1), xlabel="probability", ylabel="rejection rate", 
            title="Rejection rate vs probability", label = "direct")
# emp = param_emp(p, s)
# perm = param_perm(s)
# distances = sqrt(n/2)*sample_distances(wass, q, q, n, perm)
# histogram(distances,bins = 30)


# d1 = sqrt(n/2)*sample_distances(wass, q, p, n, perm)
# d2 = sqrt(n/2)*sample_distances(wass, q, q, n, perm)
# h1,h2 = histogram(d1, bins = 30),histogram(d2,bins = 30)
# plot(h1,h2,layout=(1,2))






Random.seed!(1234)
d_w, d_l = zeros(20), zeros(20)
for i in 1:20
    p_emp, q_emp = generate_emp(p, 5, 5), generate_emp(p, 5, 5)
    d_w[i] = ww(p_emp, q_emp)
    d_l[i] = dlip(p_emp, q_emp)
end

pl = plot()
scatter!(pl, d_w, label = "wass")
scatter!(pl, d_l, label = "dlip")

Random.seed!(1234)
d_w_s = sample_distances(ww, p, q, 10, 10, s)
Random.seed!(1234)
d_l_s = sample_distances(dlip, p, q, 10, 10, s)
pl_s = plot()
scatter!(pl_s, d_w_s, label = "wass")
scatter!(pl_s, d_l_s, label = "dlip")
