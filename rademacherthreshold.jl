using Plots


include("distributions.jl")
include("structures.jl")
include("distances/w_distance.jl")
include("distances/new_distance.jl")
include("distances/distance_Wasserstein.jl")









# given two laws of random probability measures, generate distances between hierarchical
# empirical Measures

function sample_distances(q_1::PPM, q_2::PPM, n::Int, m::Int, s::Int)
    # this function samples distances between hierarchical empirical measures
    # n : number of probability measures from which observations are generated
    # m : number of observations from each probability measure
    # s : number of times to sample distance between empirical measures

    d_wws = Vector{Float64}(undef, s)
    d_lips = Vector{Float64}(undef, s)
    for i in 1:s
        emp_1, emp_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        d_wws[i] = ww(emp_1, emp_2)
        d_lips[i] = dlip(emp_1, emp_2)
    end
    return (d_wws, d_lips)
end


function c_1(x::Float64, n::Int, K::Float64)
    a = 1280 * K * √2 / √n 
    b = sqrt(4 * K^2 * log(1/x) / n)
    c = 4 * K * log(1/x)/(3 * n)
    d = 32 * sqrt(10 * K^2 * log(1/x) * sqrt(log(2)))/ (n ^ (3/4))
    return a + b + c + d
end

function c_2(x::Float64, m::Int64, K::Float64)
    a = 256 * K * sqrt(log(2)) / √m
    b = sqrt(4 * K^2 * log(1/x) / m)
    c = 4 * K * log(1/x)/(3 * m)
    d = 64 * sqrt(2 * K^2 * log(1/x) * sqrt(log(2)))/ (m ^ (3/4))
    return a + b + c + d
end
    





function rademacher_threshold(n::Int, m::Int, θs::Vector{Float64}, K::Float64)
    thresholds = zeros(length(θs), 2) # first column stores thresholds for WoW, second - dlip
    for i in 1:length(θs)
        thresholds[i, 2] = 2 * (2 * c_2(θs[i] / (2 * n + 1),m, K) + c_1(θs[i] / (2 * n + 1),n, K) ) 
    end
    return thresholds
end



function permutation_threshold(n::Int, m::Int, θs::Vector{Float64})
    return zeros(length(θs), 2)
end

# given two random probability measures, get the rejection rates

function rej_rate(q_1::PPM, q_2::PPM, n::Int, m::Int, θs::Vector{Float64}, s::Int, K::Float64)
    # this function computes the rejection rate for two random probability measures
    # n : number of probability measures from which observations are generated
    # m : number of observations from each probability measure
    # θs : vector of probability levels for which we get thresholds
    # s : number of times to simulate problem

    d_wws, d_lips = sample_distances(q_1, q_2, n, m, s)
    # 2 thresholds using per each θ and distance functions(wow, hipm), one from Rademacher
    # complexity and another from permutation approach
    radem_thresh = rademacher_threshold(n, m, θs, K) # size zeros(length(θs), 2) because we have 2 distance functions
    perm_thresh = permutation_threshold(n, m, θs) # size zeros(length(θs), 2) because we have 2 distance functions

    rej_rate_wow = zeros(length(θs), 2) # for each threshold 
    rej_rate_dlips = zeros(length(θs), 2) # for each threshold
    for i in 1:length(θs)
        rej_rate_wow[i, 1] = mean(d_wws .> radem_thresh[i, 1]) # rejection rate for Rademacher complexity
        # threshold using wow
        rej_rate_wow[i, 2] = mean(d_wws .> perm_thresh[i, 1]) # rejection rate for permutation threshold using
        # wow
        rej_rate_dlips[i, 1] = mean(d_wws .> radem_thresh[i, 2]) # rejection rate for Rademacher complexity
        # threshold using dlips 
        rej_rate_dlips[i, 2] = mean(d_lips .> perm_thresh[i, 2]) # rejection rate for permutation approach 
        # threshold using dlips    
    end
    return (rej_rate_wow, rej_rate_dlips)
end



function distances_vs_rade_thresholds(q_1::PPM, q_2::PPM, n::Int, m::Int, θs::Vector{Float64}, s::Int, K::Float64)
    # this function computes the distances between two random probability measures and the Rademacher
    # thresholds
    d_wws, d_lips = sample_distances(q_1, q_2, n, m, s)
    radem_thresh = rademacher_threshold(n, m, θs, K)
    return (d_wws, d_lips, radem_thresh)
end


function empirical_vs_rade_thresholds(q_1::PPM, q_2::PPM, n::Int, m::Int, θs::Vector{Float64}, s::Int, K::Float64)
    radem_thresh = rademacher_threshold(n, m, θs, K)
    # now simulate s times random variable d(Q_{n,m}^1, Q_{n,m}^2) for both distances
    d_wws, d_lips = zeros(s), zeros(s)
    for i in 1:s
        q_emp_1, q_emp_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        d_wws[i], d_lips[i] = ww(q_emp_1, q_emp_2), dlip(q_emp_1, q_emp_2)
    end
    emp_thresh_wow, emp_thresh_dlips = quantile(d_wws, 1 .- θs), quantile(d_lips, 1 .- θs)
    return (emp_thresh_wow, emp_thresh_dlips, radem_thresh)
end

    

# I want to define discrete random probability measures and compute distances between them



q_discr_1 = discrrpm(10,10,0.0,1.0) 
q_discr_2 = discrrpm(10,10,0.0,1.0)

s = 100 # number of times to sample distances between hierarchical empirical measurse
n = 5 
m = 3
θs = collect(0.0:0.01:1.0) # probability levels for which we get thresholds

emp_thresh_wow, emp_thresh_dlips, radem_thresh = empirical_vs_rade_thresholds(q_discr_1, q_discr_2, n, m, θs, s, 1.0)


radem_thresh_wow, radem_thresh_dlips = radem_thresh[:, 1], radem_thresh[:, 2]

# plot empirical thresholds and rademacher thresholds
plot_wow = plot(xlabel  = "Probability level", ylabel = "Threshold value", title = "Empirical vs Rademacher thresholds for WoW distance (log scale)")
plot!(plot_wow, θs, log.(emp_thresh_wow), label = "Empirical WoW threshold", color = :blue)
plot!(plot_wow, θs, log.(radem_thresh_wow), label = "Rademacher WoW threshold", color = :red)
plot_dlips = plot(xlabel  = "Probability level", ylabel = "Threshold value", title = "Empirical vs Rademacher thresholds for dlip distance (log scale)")
plot!(plot_dlips, θs, log.(emp_thresh_dlips), label = "Empirical dlip threshold", color = :blue)
plot!(plot_dlips, θs, log.(radem_thresh_dlips), label = "Rademacher dlip threshold", color = :red)
plot(plot_wow, plot_dlips, layout = (1,2), size = (800,400))
# plot the rejection rates for WoW and dlip distances





plot_empiricals = plot(xlabel = "Probability level", ylabel = "Threshold value", title = "Empirical thresholds for WoW and dlip")
plot!(plot_empiricals, θs, emp_thresh_wow, label = "Empirical WoW threshold", color = :blue)
plot!(plot_empiricals, θs, emp_thresh_dlips, label = "Empirical dlip threshold", color = :red)


#rej_wow, rej_dlips = rej_rate(q_discr_1, q_discr_1, n, m, θs, s, 1.0)
#d_wws, d_lips, radem_thresh = distances_vs_rade_thresholds(q_discr_1, q_discr_1, n, m, θs, s, 1.0)


# plot d_lips distances and rademacher distances_vs_rade_thresholds
#plot(log.(d_lips), label = "dlips distances", xlabel = "sample number", ylabel = "value", title = "Distances between empirical measures")
#plot!(log.(radem_thresh[:, 2]), label = "Rademacher threshold", xlabel = "sample number", ylabel = "value", title = "Distances between empirical measures")
# plot d_wws distances and rademacher distances_vs_rade_thresholds

# d_wws = Vector{Float64}(undef, s)
# d_lips = Vector{Float64}(undef, s)
# for i in 1:s
#     emp_1, emp_2 = generate_emp(q_discr_1, 100, 1), generate_emp(q_discr_2, 100, 1)
#     d_wws[i] = ww(emp_1, emp_2)
#     d_lips[i] = dlip(emp_1, emp_2)
# end
# plot(d_wws.-d_lips, label = "WW-HIPM", xlabel = "sample number", ylabel = "value", title = "WW distance between empirical measures")



# emp_1 = generate_emp(q_discr_1, 100, 1)
# emp_2 = generate_emp(q_discr_2, 100, 1)

# d_ww = ww(emp_1, emp_2)
# d_lip = dlip(emp_1, emp_2)

# print("Distance WW: ", d_ww, "\n")
# print("Distance dlip: ", d_lip, "\n")