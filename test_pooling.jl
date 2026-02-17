include("methods.jl")



# cp denotes classical pooling
# np denotes new pooling


function decision_np(hier_sample_1::HierSample, hier_sample_2::HierSample, θ::Float64, n_permutations::Int)
    pooled_1 = vec(hier_sample_1.atoms)
    pooled_2 = vec(hier_sample_2.atoms)
    observed_distance = wasserstein_1d_equal(sort(pooled_1), sort(pooled_2))

    # get threshold
    pooled = vcat(pooled_1, pooled_2)
    n_1 = length(pooled_1)
    n_2 = length(pooled_2)
    n_total = n_1 + n_2

    samples = Vector{Float64}(undef, n_permutations)
    for i in 1:n_permutations
        random_indices = randperm(n_total) # indices to distribute rows to new hierarchical meausures
        new_pooled_1 = view(pooled, random_indices[1:n_1])
        new_pooled_2 = view(pooled, random_indices[n_1+1:n_total])
        samples[i] = wasserstein_1d_equal(sort(new_pooled_1), sort(new_pooled_2))
    end
    p_value = mean(samples .>= observed_distance)
    return Float64(p_value < θ)
end
function rejection_rate_cp_np(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, S::Int, θ::Float64,
                     n_permutations::Int)
    r_cp = 0.0
    r_np = 0.0

    @floop ThreadedEx() for s in 1:S
        hier_sample_1, hier_sample_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
        @reduce r_cp .+= decision_hipm(hier_sample_1, hier_sample_2, θ, n_permutations, false)
        @reduce r_np += decision_np(hier_sample_1, hier_sample_2, θ, n_permutations)
    end
    return r_cp / S, r_np / S
end

q_1 = DP(1.0, Beta(1,13))
q_2 = DP(1.0, Beta(1,13))  
n, m = 100, 100
n_permutations = 100
S = 100
θ = 0.05
t = time()
r_cp, r_np = rejection_rate_cp_np(q_1, q_2, n, m, S, θ,n_permutations)
dur = time() - t
println("r_cp: $(r_cp)")
println("r_np: $(r_np)")

