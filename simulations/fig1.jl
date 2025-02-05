include("../approaches/emp_threshold_approach.jl")
include("../approaches/permutation_approach.jl")
include("../distances/distance_Wasserstein.jl")
include("../distances/new_distance.jl")
include("../distances/w_distance.jl")
using QuadGK



p_1 = () -> rand(Uniform(-1/2,1/2))
function p_2()
    # with probability 1/2 return a sample from Uniform(-1,3/4) and 
    # with probability 1/2 return a sample from Uniform(3/4,1)
    if rand() < 1/2
        return rand(Uniform(-1,-3/4))
    else
        return rand(Uniform(3/4,1))
    end
end



function cdf_1(x)
    if x < -1/2
        return 0
    elseif x < 1/2
        return (x + 1/2)
    else
        return 1
    end
end

function cdf_2(x)
    if x < -1
        return 0
    elseif x < -3/4
        return 2*(x+1)
    elseif x < 3/4
        return 1/2
    elseif x < 1
        return 1/2 + + 2*(x - 3/4)
    else
        return 1
    end
end


d_true = quadgk(x -> abs(cdf_1(x) - cdf_2(x)), -1, 1)[1]


n_bottom = 5000
n_tops = [16,32,64,128,256]
s = 24
function values_fig_1(n_tops, n_bottom, s, str)
    if str == "diff"
        α = 1.0
        dp_1, dp_2 = DP(α, p_1, -1.0, 1.0), DP(α, p_2, -1.0, 1.0)
    else
        α = 1.0
        p = () -> rand()
        dp_1, dp_2 = DP(α, p, 0.0, 1.0), DP(α, p, 0.0, 1.0)
    end

    d_ww, d_lip, lower_bounds = zeros(length(n_tops)), zeros(length(n_tops)), zeros(length(n_tops))
    for i in 1:length(n_tops)
        n_top = n_tops[i]
        println("n_top: ", n_top)
        for j in 1:s
            if j % 10 == 0
                println("j: ", j)
            end
            emp_p, emp_q = generate_emp(dp_1, n_top, n_bottom), generate_emp(dp_2, n_top, n_bottom)
            d_ww[i] += ww(emp_p, emp_q)
            d_lip[i] += dlip(emp_p, emp_q)
            lower_bounds[i] += lower_bound(emp_p, emp_q)
        end
        d_ww[i] /= s
        d_lip[i] /= s
        lower_bounds[i] /= s
    end
    return d_ww, d_lip, lower_bounds
end




d_trues = [d_true for i in 1:length(n_tops)]
d_ww, d_lip, lower_bounds = values_fig_1(n_tops, n_bottom, s, "diff")
pl_1 = plot(title = "same", xlabel = "n", ylabel = "distance")
plot!(pl_1, n_tops, d_ww, seriestype=:line, marker=:circle, label="ww", color = "red")
plot!(pl_1, n_tops, d_lip, seriestype=:line, marker=:circle, label="dlip", color = "blue")
plot!(pl_1, n_tops, d_trues, seriestype=:line, marker=:circle, label="true", color = "black")
plot!(pl_1, n_tops, lower_bounds, seriestype=:line, marker=:circle, label="lower bound", color = "green")


d_ww, d_lip, lower_bounds = values_fig_1(n_tops, n_bottom, s, "same")
pl_2 = plot(title = "different", xlabel = "n", ylabel = "distance")
plot!(pl_2, n_tops, d_ww, seriestype=:line, marker=:circle, label="ww", color = "red")
plot!(pl_2, n_tops, d_lip, seriestype=:line, marker=:circle, label="dlip", color = "blue")
plot!(pl_2, n_tops, lower_bounds, seriestype=:line, marker=:circle, label="lower bound", color = "green")









# filepath = joinpath(pwd(),"plots")
# savefig(pl_1, joinpath(filepath, "fig_1_left"))
# savefig(pl_2, joinpath(filepath, "fig_1_right"))