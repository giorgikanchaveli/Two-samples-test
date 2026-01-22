using RCall # to call R functions
using Plots

include("distributions.jl")

include("distances/new_distance.jl")
include("distances/distance_Wasserstein.jl")
using DataFrames
using CSV
using FLoops
using QuadGK

function rejection_rate_hipm_permutation_parallel(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)
    rej_rate = 0.0

    @floop ThreadedEx() for s in 1:S
        hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        a = minimum([hier_sample_1.a, hier_sample_2.a])
        b = maximum([hier_sample_1.b, hier_sample_2.b])
        hier_sample_1.a = a
        hier_sample_2.a = a
        hier_sample_1.b = b
        hier_sample_2.b = b
        observed_test_stat = dlip(hier_sample_1, hier_sample_2)
        
        # obtain quantile using permutation approach
        permutation_samples = zeros(n_permutation) # zeros can be improved
        
        total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
        for i in 1:n_permutation
            random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

            atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
            atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
            hier_sample_1_permutation = HierSample(atoms_1, a, b)
            hier_sample_2_permutation = HierSample(atoms_2, a, b)

            permutation_samples[i] = dlip(hier_sample_1_permutation, hier_sample_2_permutation)
        end
        threshold = quantile(permutation_samples, 1 - θ)
        
        @reduce rej_rate += 1.0*(observed_test_stat > threshold)
    end
    return rej_rate / S
end

function rejection_rate_hipm_permutation_wrong(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, S::Int, θ::Float64, n_permutation::Int)

    # firstly we obtain threshold
    hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
    a = minimum((hier_sample_1.a, hier_sample_2.a))
    b = maximum((hier_sample_1.b, hier_sample_2.b))
    hier_sample_1.a = a
    hier_sample_2.a = a
    hier_sample_1.b = b
    hier_sample_2.b = b

    permutation_samples = zeros(n_permutation) # zeros can be improved
    total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
        for i in 1:n_permutation
            random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

            atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
            atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
            hier_sample_1_permutation = HierSample(atoms_1, a, b)
            hier_sample_2_permutation = HierSample(atoms_2, a, b)

            permutation_samples[i] = dlip(hier_sample_1_permutation, hier_sample_2_permutation)
        end
    threshold = quantile(permutation_samples, 1 - θ)

    rej_rate = 0.0

    @floop ThreadedEx() for s in 1:S
        local hier_sample_1 = generate_emp(q_1, n, m)
        local hier_sample_2 = generate_emp(q_2, n, m)
        local a = minimum([hier_sample_1.a, hier_sample_2.a])
        local b = maximum([hier_sample_1.b, hier_sample_2.b])
        hier_sample_1.a = a
        hier_sample_2.a = a
        hier_sample_1.b = b
        hier_sample_2.b = b
        observed_test_stat = dlip(hier_sample_1, hier_sample_2)
        
        @reduce rej_rate += 1.0*(observed_test_stat > threshold)
    end
    return rej_rate / S
end


function wass_beta(p_1::Beta{Float64}, p_2::Beta{Float64})
    function f(x::Float64)
       
        return abs(cdf(p_1, x) - cdf(p_2,x))
    end
    return quadgk(f, 0.0, 1.0)[1]

end


function J_1(dist::Beta{Float64})
    function f(x::Float64)
        c = cdf(dist, x)
        return sqrt(c*(1 - c))
    end
    return quadgk(f, 0.0, 1.0)[1]
end

println(Threads.nthreads())
α = 0.2
p_1 = Beta(1.0,1.0)
p_2 = Beta(1.0,1.3) 

wass_beta(p_1,p_2)


m_star = ( (   (sqrt(2) + 1 / sqrt(2)) * (J_1(p_1) * sqrt(α) / sqrt(α + 1))     )/   ((sqrt(2)-1)*wass_beta(p_1,p_2))   )^2

q_1 = DP(α,p_1, 0.0, 1.0)
q_2 = DP(α,p_2, 0.0, 1.0)

ms = [2^i for i in 1:10]
ns = [100, 200]

S = 32*13
n_permutation = 100
θ = 0.05

rej_rates = zeros(length(ns), length(ms))
t = time()
for i in 1:length(ns)
    for j in 1:length(ms)
        rej_rates[i, j] = rejection_rate_hipm_permutation_parallel(q_1, q_2, ns[i], ms[j], S, θ, n_permutation)
    end
end
duration = time() - t
rej_rates
println("duration in hours is $(duration / 3600)" )

col_names = Symbol.("m = " .* string.(ms))
df = DataFrame(rej_rates, col_names)
df = hcat(DataFrame("n" => "n = " .* string.(ns)), df)
CSV.write("n_vs_m_hipm/n_vs_m_hipm_S=$(S)_n_permutation=$(n_permutation).csv", df)


# define two DP
# get m*
# check rejection rates for m < m* and m > m* for fixed n = 100
