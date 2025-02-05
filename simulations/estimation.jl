include("../approaches/emp_threshold_approach.jl")
include("../approaches/permutation_approach.jl")
include("../distances/distance_Wasserstein.jl")
include("../distances/new_distance.jl")
include("../distances/w_distance.jl")
using QuadGK



# generate measures -> compute 2 distances

function int_wass_beta(p_a, p_b, q_a, q_b)
    # this function computes the Wasserstein distance between two Beta distributions
    # with parameters p and q using numerical integration
    p = Beta(p_a, p_b)
    q = Beta(q_a, q_b)
    cdf_p = x -> cdf(p, x)
    cdf_q = x -> cdf(q, x)
    f = x -> abs(cdf_p(x) - cdf_q(x))
    return quadgk(f, 0, 1)[1]
    return wass(p, q)
end


function mc_wass_beta(p_a, p_b, q_a, q_b, n, s)
    # it is slow and not very precise
    # this function computes the Wasserstein distance between two Beta distributions
    # with parameters p and q using monte carlo 
    p = Beta(p_a, p_b)
    q = Beta(q_a, q_b)
    d = zeros(s)
    for i in 1:s
        d[i] = wass(rand(p, n), rand(q, n))
    end
    return mean(d)
end

function get_distance(p::PPM, q::PPM, n::Int, m::Int, s::Int)
    # this function computes the distances between the empirical measures
    # sampled directly from p and q

    # s : number of samples
    # n,m : n_top, n_bottom

    d_ww, d_lip = zeros(s), zeros(s)

    for i in 1:s
        if i % 10 == 0
            println("sample: ", i)
        end
        p_emp, q_emp = generate_emp(p, n, m), generate_emp(q, n, m)
        d_ww[i] = ww(p_emp, q_emp)
        d_lip[i] = dlip(p_emp, q_emp)
    end
    return d_ww, d_lip
end


# Define measures

# p_1 = ()->probability("same")
# p_2 = ()->probability("splitting")
# dp_1 = DP(1.0, p_1, -1.0, 1.0)
# dp_2 = DP(1.0, p_2, -1.0, 1.0)
p_a, p_b = 1.2, 0.3
q_a, q_b = 1.1, 3.4
beta_p, beta_q = Beta(p_a, p_b), Beta(q_a, q_b)
p_1 = ()->rand(beta_p)
p_2 = ()->rand(beta_q)
α = 1.0
dp_1 = DP(α, p_1, 0.0, 1.0)
dp_2 = DP(α, p_2, 0.0, 1.0)
n_top, n_bottom = 50, 10

Random.seed!(1234)
# compute distances using direct samples
s = 24
d_ww, d_lip = get_distance(dp_1, dp_2, n_top, n_bottom, s)
#d_mc = mc_wass_beta(p_a, p_b, q_a, q_b, 1000000,100)
d_int = round(int_wass_beta(p_a, p_b, q_a, q_b), digits = 6)


summary = plot(title = "summary")
sc = plot(title = "Test statistics, ww vs dlip", xlabel = "sample", ylabel = "distance")
scatter!(sc, d_ww, label = "ww", color = "red")
scatter!(sc, d_lip, label = "dlip", color = "blue")
hline!(sc, [mean(d_ww)], label="mean ww", color="red")
hline!(sc, [mean(d_lip)], label="mean dlip", color="blue")
hline!(sc, [d_int], label="int_estimate", color="green")
var_ww = round(var(d_ww), digits=6)
var_lip = round(var(d_lip), digits=6)
scatter!(summary, [], [], label="Var_ww = $(var_ww)")  # insert text variance for ww
scatter!(summary, [], [], label="Var_dlip = $(var_lip)")  # insert text variance for dlip
scatter!(summary, [], [], label="d_int = $(d_int)")  # insert text for int_estimate of distance*sqrt(n/2)

bias_ww = round(mean(d_ww) - d_int, digits=6)
bias_lip = round(mean(d_lip) - d_int, digits=6)
scatter!(summary, [], [], label="Bias_ww = $(bias_ww)")  # insert text bias for ww
scatter!(summary, [], [], label="Bias_dlip = $(bias_lip)")  # insert text bias for dlip


# save figures

# filepath = joinpath(pwd(),"plots/n = $(n_top), m = $(n_bottom)")
# savefig(sc, joinpath(filepath, "estimation_ww_lip_$(n_top)_$(n_bottom)"))
# savefig(summary, joinpath(filepath, "summary_ww_lip_$(n_top)_$(n_bottom)"))