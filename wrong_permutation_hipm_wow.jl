using Plots

include("distributions.jl")

include("distances/new_distance.jl")
include("distances/distance_Wasserstein.jl")

using FLoops

# We test if using proper permutation approach has significantly better performance than using wrong approach (fixing threshold from one pair of hierarchical samples).


# Let us firstly do for HIPM
function permutation_threshold_hipm(hier_sample_1::emp_ppm, hier_sample_2::emp_ppm, θ::Float64, n_permutations::Int)
    n = hier_sample_1.n
    m = hier_sample_1.m
    a = minimum((hier_sample_1.a, hier_sample_2.a))
    b = maximum((hier_sample_1.b, hier_sample_2.b))
    hier_sample_1.a = a
    hier_sample_2.a = a
    hier_sample_1.b = b
    hier_sample_2.b = b

    permutation_samples = zeros(n_permutations) # zeros can be improved

    total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
        for i in 1:n_permutations
            random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

            atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
            atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
            hier_sample_1_permutation = emp_ppm(atoms_1, n, m, a, b)
            hier_sample_2_permutation = emp_ppm(atoms_2, n, m, a, b)

            permutation_samples[i] = dlip(hier_sample_1_permutation, hier_sample_2_permutation)
        end
    return quantile(permutation_samples, 1 - θ)
end

function rejection_rate_hipm(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_permutations::Int)
    # firstly we obtain threshold for wrong approach

    threshold_wrong = permutation_threshold_hipm(generate_emp(q_1,n,m), generate_emp(q_2, n, m), θ, n_permutations)
    
    rej_rate_wrong = 0.0
    rej_rate_proper = 0.0

    @floop ThreadedEx() for s in 1:S
        hier_sample_1 = generate_emp(q_1, n, m)
        hier_sample_2 = generate_emp(q_2, n, m)

        threshold_proper = permutation_threshold_hipm(hier_sample_1, hier_sample_2, θ, n_permutations) # obtain proper threshold
        # note that a and b for those samples are set by permutation_threshold_hipm function.

        observed_test_stat = dlip(hier_sample_1, hier_sample_2)
        
        @reduce rej_rate_wrong += 1.0*(observed_test_stat > threshold_wrong) # decision using wrong threshold
        @reduce rej_rate_proper += 1.0*(observed_test_stat > threshold_proper) # decision using proper threshold
    end
    rej_rate_wrong /= S
    rej_rate_proper /= S
    return (rej_rate_wrong, rej_rate_proper)
end


function save_hipm(n::Int, m::Int, S::Int, θ::Float64, n_permutations::Int, βs::Vector{Float64})
    # for differt β parameter for Beta distribution, we obtain the rejection rates.

    n_pairs = length(βs) # total number of pairs of laws of RPMs considered

    rej_rates_wrong = zeros(n_pairs) 
    rej_rates_proper = zeros(n_pairs)
    
    # We only change p_2 for second Dirichlet process
    α = 1.0
    p_1 = Beta(1.0,1.0)
    q_1 = DP(α, p_1, 0.0, 1.0)
    
    for i in 1:n_pairs
        p_2 = Beta(1.0, βs[i])
        q_2 = DP(α, p_2, 0.0, 1.0)

        rej_rates = rejection_rate_hipm(q_1, q_2, n, m, S, θ, n_permutations)
        rej_rates_wrong[i] = rej_rates[1]
        rej_rates_proper[i] = rej_rates[2]
    end
    rej_plot = plot(title = "Rej rates wrong vs proper threshold, hipm", xlabel = "β", ylabel = "Rej rate", 
                                        xlims=(0.9, 2.1), ylims = (-0.1, 1.1))
    plot!(rej_plot, βs, rej_rates_proper, label = "proper", color = "green",marker = (:circle, 4))
    plot!(rej_plot, βs, rej_rates_wrong, label = "wrong", color = "red", marker = (:circle, 4))

    filepath = joinpath(pwd(), "plots/wrong_vs_proper")
    savefig(rej_plot,joinpath(filepath, "hipm_wrong_vs_wrong_n=$(n)_m=$(m)_S=$(S)_npermutation=$(n_permutations).png"))
end

# Now everything same but for WoW




function permutation_threshold_wow(hier_sample_1::emp_ppm, hier_sample_2::emp_ppm, θ::Float64, n_permutations::Int)
    n = hier_sample_1.n
    m = hier_sample_1.m
    a = minimum((hier_sample_1.a, hier_sample_2.a))
    b = maximum((hier_sample_1.b, hier_sample_2.b))
    hier_sample_1.a = a
    hier_sample_2.a = a
    hier_sample_1.b = b
    hier_sample_2.b = b

    permutation_samples = zeros(n_permutations) # zeros can be improved

    total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
        for i in 1:n_permutations
            random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

            atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
            atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
            hier_sample_1_permutation = emp_ppm(atoms_1, n, m, a, b)
            hier_sample_2_permutation = emp_ppm(atoms_2, n, m, a, b)

            permutation_samples[i] = ww(hier_sample_1_permutation, hier_sample_2_permutation)
        end
    return quantile(permutation_samples, 1 - θ)
end

function rejection_rate_wow(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_permutations::Int)
    # firstly we obtain threshold for wrong approach

    threshold_wrong = permutation_threshold_wow(generate_emp(q_1,n,m), generate_emp(q_2, n, m), θ, n_permutations)
    
    rej_rate_wrong = 0.0
    rej_rate_proper = 0.0

    @floop ThreadedEx() for s in 1:S
        hier_sample_1 = generate_emp(q_1, n, m)
        hier_sample_2 = generate_emp(q_2, n, m)

        threshold_proper = permutation_threshold_wow(hier_sample_1, hier_sample_2, θ, n_permutations) # obtain proper threshold
        # note that a and b for those samples are set by permutation_threshold_wow function.

        observed_test_stat = ww(hier_sample_1, hier_sample_2)
        
        @reduce rej_rate_wrong += 1.0*(observed_test_stat > threshold_wrong) # decision using wrong threshold
        @reduce rej_rate_proper += 1.0*(observed_test_stat > threshold_proper) # decision using proper threshold
    end
    rej_rate_wrong /= S
    rej_rate_proper /= S
    return (rej_rate_wrong, rej_rate_proper)
end


function save_wow(n::Int, m::Int, S::Int, θ::Float64, n_permutations::Int, βs::Vector{Float64})
    # for differt β parameter for Beta distribution, we obtain the rejection rates.

    n_pairs = length(βs) # total number of pairs of laws of RPMs considered

    rej_rates_wrong = zeros(n_pairs) 
    rej_rates_proper = zeros(n_pairs)
    
    # We only change p_2 for second Dirichlet process
    α = 1.0
    p_1 = Beta(1.0,1.0)
    q_1 = DP(α, p_1, 0.0, 1.0)
    
    for i in 1:n_pairs
        p_2 = Beta(1.0, βs[i])
        q_2 = DP(α, p_2, 0.0, 1.0)

        rej_rates = rejection_rate_wow(q_1, q_2, n, m, S, θ, n_permutations)
        rej_rates_wrong[i] = rej_rates[1]
        rej_rates_proper[i] = rej_rates[2]
    end
    rej_plot = plot(title = "Rej rates wrong vs proper threshold, wow", xlabel = "β", ylabel = "Rej rate", 
                                        xlims=(0.9, 2.1), ylims = (-0.1, 1.1))
    plot!(rej_plot, βs, rej_rates_proper, label = "proper", color = "green",marker = (:circle, 4))
    plot!(rej_plot, βs, rej_rates_wrong, label = "wrong", color = "red", marker = (:circle, 4))

    filepath = joinpath(pwd(), "plots/wrong_vs_proper")
    savefig(rej_plot,joinpath(filepath, "wow_wrong_vs_wrong_n=$(n)_m=$(m)_S=$(S)_npermutation=$(n_permutations).png"))
end




println("number of threads: $(Threads.nthreads())")

βs = collect(1.0:0.1:2.0)
println("expected duration 30 hours")


n = 100
m = 100
S = 32
n_permutations = 100
θ = 0.05

t = time()
save_hipm(n, m, S, θ, n_permutations,βs)
duration_hipm = time() - t

println("duration for HIPM is $(duration_hipm/3600) hours")

t = time()
save_wow(n, m, S, θ, n_permutations,βs)
duration_wow = time() - t
println("duration for WoW is $(duration_wow/3600) hours")

total_duration = (duration_wow + duration_hipm)/3600 # hours
println("Total duration is $(total_duration) hours")

