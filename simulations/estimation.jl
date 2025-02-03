include("../approaches/emp_threshold_approach.jl")
include("../approaches/permutation_approach.jl")
include("../distances/distance_Wasserstein.jl")
include("../distances/new_distance.jl")
include("../distances/w_distance.jl")



# generate measures -> compute 2 distances

function mc_wass_beta(p_a, p_b, q_a, q_b)
    # this function computes the Wasserstein distance between two Beta distributions
    # with parameters p and q
    p = Beta(p_a, p_b)
    q = Beta(q_a, q_b)
    s = 2000
    n = 50000
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
p_a, p_b = 2.5, 4.0
q_a, q_b = 2.5, 4.0
beta_p, beta_q = Beta(p_a, p_b), Beta(q_a, q_b)
p_1 = ()->rand(beta_p)
p_2 = ()->rand(beta_q)
dp_1 = DP(2.0, p_1, -1.0, 1.0)
dp_2 = DP(1.0, p_1, -1.0, 1.0)
n_top, n_bottom = 10, 2


# compute distances using direct samples
s = 30
d_ww, d_lip = get_distance(dp_1, dp_2, n_top, n_bottom, s)
d_mc = mc_wass_beta(p_a, p_b, q_a, q_b)
t_mc = round(sqrt(n_top/2)*d_mc, digits = 3)
t_ww = sqrt(n_top/2)*d_ww
t_lip = sqrt(n_top/2)*d_lip

summary = plot(title = "summary")
sc = plot(title = "Test statistics, ww vs dlip", xlabel = "sample", ylabel = "distance")
scatter!(sc, t_ww, label = "ww", color = "red")
scatter!(sc, t_lip, label = "dlip", color = "blue")
hline!(sc, [mean(t_ww)], label="mean ww", color="red")
hline!(sc, [mean(t_lip)], label="mean dlip", color="blue")
hline!(sc, [t_mc], label="mc estimate", color="green")
var_ww = round(var(t_ww), digits=3)
var_lip = round(var(t_lip), digits=3)
scatter!(summary, [], [], label="Var_ww = $(var_ww)")  # insert text variance for ww
scatter!(summary, [], [], label="Var_dlip = $(var_lip)")  # insert text variance for dlip
scatter!(summary, [], [], label="t_mc = $(t_mc)")  # insert text for mc estimate of distance*sqrt(n/2)

bias_ww = round(mean(t_ww) - t_mc, digits=3)
bias_lip = round(mean(t_lip) - t_mc, digits=3)
scatter!(summary, [], [], label="Bias_ww = $(bias_ww)")  # insert text bias for ww
scatter!(summary, [], [], label="Bias_dlip = $(bias_lip)")  # insert text bias for dlip


filepath = joinpath(pwd(),"plots/n = $(n_top), m = $(n_bottom)")
savefig(sc, joinpath(filepath, "estimation_ww_lip_$(n_top)_$(n_bottom)"))
savefig(summary, joinpath(filepath, "summary_ww_lip_$(n_top)_$(n_bottom)"))