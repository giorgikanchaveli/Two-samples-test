include("methods.jl")



function permutation_samples_pooling(hier_sample_1::HierSample, hier_sample_2::HierSample, n_permutations::Int)
    pooled_1 = vec(hier_sample_1.atoms)
    pooled_2 = vec(hier_sample_2.atoms)
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
    return samples
end


function permutation_samples_averaging(hier_sample_1::HierSample, hier_sample_2::HierSample, n_permutations::Int)
    all_atoms = vcat(hier_sample_1.atoms, hier_sample_2.atoms)
    samples = Vector{Float64}(undef, n_permutations)
    n_1 = size(hier_sample_1.atoms, 1)
    n_2 = size(hier_sample_2.atoms, 1)
    n_total = n_1 + n_2

    for i in 1:n_permutations
        random_indices = randperm(n_total) # indices to distribute rows to new hierarchical meausures
        new_atoms_1 = view(all_atoms, random_indices[1:n_1], :) # first rows indexed by n random indices to the atoms_1
        new_atoms_2 = view(all_atoms, random_indices[n_1+1:end], :) # first rows indexed by n random indices to the atoms_2
        
        new_pooled_1 = vec(new_atoms_1)
        new_pooled_2 = vec(new_atoms_2)
        samples[i] = wasserstein_1d_equal(sort(new_pooled_1), sort(new_pooled_2))
    end
    return samples
end

function pvalue_pooling(hier_sample_1::HierSample, hier_sample_2::HierSample, n_permutations::Int)
    pooled_1 = vec(hier_sample_1.atoms)
    pooled_2 = vec(hier_sample_2.atoms)
    observed_distance = wasserstein_1d_equal(sort(pooled_1), sort(pooled_2))
    samples = permutation_samples_pooling(hier_sample_1, hier_sample_2, n_permutations)
    return mean(samples .>= observed_distance)
end

function pvalue_averaging(hier_sample_1::HierSample, hier_sample_2::HierSample, n_permutations::Int)
    pooled_1 = vec(hier_sample_1.atoms)
    pooled_2 = vec(hier_sample_2.atoms)
    observed_distance = wasserstein_1d_equal(sort(pooled_1), sort(pooled_2))

    # get threshold
    samples = permutation_samples_averaging(hier_sample_1, hier_sample_2, n_permutations)
    return mean(samples .>= observed_distance)
end

function decision_pooling(hier_sample_1::HierSample, hier_sample_2::HierSample, θ::Float64, n_permutations::Int)
    return Float64(pvalue_pooling(hier_sample_1, hier_sample_2, n_permutations)< θ)
end

function decision_averaging(hier_sample_1::HierSample, hier_sample_2::HierSample, θ::Float64, n_permutations::Int)
    return Float64(pvalue_averaging(hier_sample_1, hier_sample_2, n_permutations)< θ)
end

function rejection_rate(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, S::Int, θ::Float64,
                     n_permutations::Int)
    r_averaging = 0.0
    r_pooling = 0.0

    @floop ThreadedEx() for s in 1:S
        hier_sample_1, hier_sample_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
        @reduce r_averaging += decision_averaging(hier_sample_1, hier_sample_2, θ, n_permutations)
        @reduce r_pooling += decision_pooling(hier_sample_1, hier_sample_2, θ, n_permutations)
    end
    return r_averaging / S, r_pooling / S
end

q_1 = DP(1.0, Beta(12,3))
q_2 = DP(1.0, Beta(12,3))  
n, m = 200, 100
n_permutations = 100
S = 100
θ = 0.05
t = time()

r_averaging, r_pooling = rejection_rate(q_1, q_2, n, m, S, θ, n_permutations)





hier_sample_1 = generate_hiersample(q_1, n, m)
hier_sample_2 = generate_hiersample(q_2, n, m)

samples_pooling = permutation_samples_pooling(hier_sample_1, hier_sample_2, n_permutations)
samples_averaging = permutation_samples_averaging(hier_sample_1, hier_sample_2, n_permutations)

observed_distance = wasserstein_1d_equal(sort(vec(hier_sample_1.atoms)), sort(vec(hier_sample_2.atoms)))

# pvalue = pvalue_np(h_1, h_2, n_permutations)

# r_cp, r_np = rejection_rate_cp_np(q_1, q_2, n, m, S, θ,n_permutations)
# dur = time() - t
# println("r_cp: $(r_cp)")
# println("r_np: $(r_np)")



