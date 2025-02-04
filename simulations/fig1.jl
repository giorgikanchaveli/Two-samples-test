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
        return rand(Uniform(-1,3/4))
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



α = 1.0
dp_1, dp_2 = DP(α, p_1, -1.0, 1.0), DP(α, p_2, -1.0, 1.0)

n_bottom = 5000
n_tops = collect(10:50:500)
s = 24

d_ww, d_lip = zeros(length(n_tops)), zeros(length(n_tops))

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
    end
    d_ww[i] /= s
    d_lip[i] /= s
end

pl = plot(xlabel = "n", ylabel = "distance")
plot!(pl, n_tops, d_ww, label = "ww")
plot!(pl, n_tops, d_lip, label = "dlip")
plot!(pl, n_tops, d_true, label = "true value")
