# Checks that the Fréchet test (decision_frechet and its helpers in methods.jl) is correct:
#   1) frechet_statistic matches DenANOVAStatistic from the R package frechet (Dubey and Muller's own code),
#      and the formula as written in Dubey and Muller (2019) (pooled-variance F_n, product form of U_n).
#   2) Algebraic identities and invariances of T_n.
#   3) Input validation and degenerate statistics throw informative errors.
#   4) p-values and decisions have the right form, and the test has correct size and power in a small simulation.
#
# DenANOVAStatistic takes quantile functions on a grid and integrates them with the trapezoidal rule. To pass it the
# step quantile function of a row with m atoms, we put two grid points ε apart on each step, so the trapezoidal rule
# is exact up to O(ε).

using Test
using Random
using RCall

include("../methods.jl")


statistic(atoms_1, atoms_2) = frechet_statistic(atoms_1, atoms_2, "test samples")


# Statistic as written in Dubey and Muller (2019): F_n = V_pooled - Σ λ_j V_j, U_n = λ_1 λ_2 (V_1 - V_2)^2 / (σ2_1 σ2_2).
function test_statistic_paper(atoms_1, atoms_2)
    n_1, n_2 = size(atoms_1)[1], size(atoms_2)[1]
    n_total = n_1 + n_2
    λ_1, λ_2 = n_1 / n_total, n_2 / n_total

    distances_1 = squared_distances_to_mean(atoms_1)
    distances_2 = squared_distances_to_mean(atoms_2)
    distances_pooled = squared_distances_to_mean(vcat(atoms_1, atoms_2))
    V_1, V_2, V_pooled = mean(distances_1), mean(distances_2), mean(distances_pooled)
    σ2_1 = mean(distances_1.^2) - V_1^2
    σ2_2 = mean(distances_2.^2) - V_2^2

    F_n = V_pooled - λ_1 * V_1 - λ_2 * V_2
    U_n = λ_1 * λ_2 * (V_1 - V_2)^2 / (σ2_1 * σ2_2)
    return n_total * U_n / (λ_1 / σ2_1 + λ_2 / σ2_2) + n_total * F_n^2 / (λ_1^2 * σ2_1 + λ_2^2 * σ2_2)
end


# Statistic computed by frechet::DenANOVAStatistic in R.
function test_statistic_r(atoms_1, atoms_2)
    m = size(atoms_1)[2]
    ε = 1e-10
    grid = sort(vcat([(k - 1) / m + ε for k in 1:m], [k / m for k in 1:m])) # two grid points on each step
    rows = vcat(atoms_1, atoms_2)
    qin = [repeat(rows[i, :], inner = 2) for i in 1:size(rows)[1]]
    sizes = [size(atoms_1)[1], size(atoms_2)[1]]
    @rput qin grid sizes
    R"tn <- frechet:::DenANOVAStatistic(qin, 1:length(qin), sizes, grid)"
    return rcopy(R"tn")
end


function create_test_cases()
    laws = [(tnormal_normal(0.0, 0.5, -10.0, 10.0), tnormal_normal(0.3, 0.5, -10.0, 10.0)),
            (normal_normal_B(0.0, 1.0, 1.0), normal_normal_B(0.0, 1.0, 2.0)),
            (DP(1.0, Uniform(0.0, 1.0)), DP(1.0, Beta(2.0, 2.0))), # rows with repeated atoms
            (discr_normal([-1.0, 1.0], [0.5, 0.5]), discr_normal([-2.0, 0.0, 2.0], [1/8, 3/4, 1/8]))]
    sizes = [(20, 20, 30), (5, 40, 30), (40, 3, 10), (15, 25, 1)] # (n_1, n_2, m), including unequal group sizes

    test_cases = Tuple{HierSample, HierSample}[]
    for (q_1, q_2) in laws, (n_1, n_2, m) in sizes
        push!(test_cases, (generate_hiersample(q_1, n_1, m), generate_hiersample(q_2, n_2, m)))
    end
    return test_cases
end


Random.seed!(1234)
test_cases = create_test_cases()


@testset "Fréchet test" begin

    @testset "Statistic matches R package frechet and the paper's formula" begin
        for (h_1, h_2) in test_cases
            t = statistic(h_1.atoms, h_2.atoms)
            @test isapprox(t, test_statistic_r(h_1.atoms, h_2.atoms); rtol = 1e-8)
            @test isapprox(t, test_statistic_paper(h_1.atoms, h_2.atoms); rtol = 1e-10)
        end
    end

    @testset "Squared distances and Fréchet mean" begin
        atoms = [1.0 2.0 3.0; 3.0 4.0 7.0] # Fréchet mean has atoms [2.0, 3.0, 5.0]
        @test squared_distances_to_mean(atoms) ≈ [(1 + 1 + 4) / 3, (1 + 1 + 4) / 3]

        # rows that are translates of each other: W2 distance is the size of the translation
        atoms = [0.0 1.0 5.0; 2.0 3.0 7.0; 4.0 5.0 9.0]
        @test squared_distances_to_mean(atoms) ≈ [4.0, 0.0, 4.0]
    end

    @testset "Direct F_n equals pooled variance minus within-group variances" begin
        for (h_1, h_2) in test_cases
            atoms_1, atoms_2 = h_1.atoms, h_2.atoms
            n_1, n_2 = size(atoms_1)[1], size(atoms_2)[1]
            λ_1, λ_2 = n_1 / (n_1 + n_2), n_2 / (n_1 + n_2)

            F_direct = λ_1 * λ_2 * mean((mean(atoms_1, dims = 1) .- mean(atoms_2, dims = 1)).^2)
            F_pooled = mean(squared_distances_to_mean(vcat(atoms_1, atoms_2))) -
                       λ_1 * mean(squared_distances_to_mean(atoms_1)) - λ_2 * mean(squared_distances_to_mean(atoms_2))
            @test isapprox(F_direct, F_pooled; rtol = 1e-10, atol = 1e-14)
        end
    end

    @testset "Invariances of T_n" begin
        for (h_1, h_2) in test_cases
            atoms_1, atoms_2 = h_1.atoms, h_2.atoms
            t = statistic(atoms_1, atoms_2)

            @test statistic(atoms_2, atoms_1) ≈ t # swapping groups
            @test statistic(atoms_1[randperm(size(atoms_1)[1]), :], atoms_2) ≈ t # reordering rows
            @test statistic(atoms_1 .+ 3.0, atoms_2 .+ 3.0) ≈ t # translation of all atoms
            @test statistic(2.5 .* atoms_1, 2.5 .* atoms_2) ≈ t # scaling of all atoms
        end
    end

    @testset "Group with two rows (σ2 = 0 for that group only)" begin
        # With two rows, both rows are at the same distance from their Fréchet mean, so σ2_1 = 0. The product form of
        # U_n (used in R and in the paper) is then 0/0, while T_n has the limit below.
        atoms_1 = generate_hiersample(tnormal_normal(0.0, 0.5, -10.0, 10.0), 2, 30).atoms
        atoms_2 = generate_hiersample(tnormal_normal(0.0, 0.5, -10.0, 10.0), 30, 30).atoms
        n_1, n_2 = size(atoms_1)[1], size(atoms_2)[1]
        n_total = n_1 + n_2
        λ_1, λ_2 = n_1 / n_total, n_2 / n_total

        distances_2 = squared_distances_to_mean(atoms_2)
        V_1, V_2 = mean(squared_distances_to_mean(atoms_1)), mean(distances_2)
        σ2_2 = mean((distances_2 .- V_2).^2)
        F_n = λ_1 * λ_2 * mean((mean(atoms_1, dims = 1) .- mean(atoms_2, dims = 1)).^2)
        t_limit = (V_1 - V_2)^2 * n_2 / σ2_2 + n_total * F_n^2 / (λ_2^2 * σ2_2)

        t = statistic(atoms_1, atoms_2)
        @test isfinite(t)
        @test isapprox(t, t_limit; rtol = 1e-10)
        @test !isfinite(test_statistic_r(atoms_1, atoms_2)) # R's product form breaks down
    end

    @testset "Input validation" begin
        atoms = generate_hiersample(tnormal_normal(0.0, 0.5, -10.0, 10.0), 10, 30).atoms
        h = HierSample(atoms, minimum(atoms), maximum(atoms))
        h_one_row = HierSample(atoms[1:1, :], minimum(atoms), maximum(atoms))
        atoms_nan = copy(atoms); atoms_nan[2, 5] = NaN
        atoms_inf = copy(atoms); atoms_inf[2, 5] = Inf

        @test check_atoms_frechet(atoms, atoms) === nothing
        @test_throws ArgumentError check_atoms_frechet(atoms[1:1, :], atoms) # one row
        @test_throws ArgumentError check_atoms_frechet(atoms, atoms[:, 1:20]) # different number of columns
        @test_throws ArgumentError check_atoms_frechet(zeros(3, 0), zeros(3, 0)) # no columns
        @test_throws ArgumentError check_atoms_frechet(atoms_nan, atoms)
        @test_throws ArgumentError check_atoms_frechet(atoms, atoms_inf)
        @test_throws ArgumentError pvalue_frechet(h_one_row, h, 100, true)
        @test_throws ArgumentError pvalue_frechet(h, h, 0, true)
        @test_throws ArgumentError decision_frechet(h, h, 0.05, 0, false)
    end

    @testset "Degenerate statistics throw DomainError" begin
        # Small integer/dyadic atoms, so that Fréchet means and distances are computed without rounding and σ2 = 0 exactly.

        # both groups have two rows
        two_1 = [0.0 1.0 2.0; 2.0 3.0 6.0]
        two_2 = [1.0 1.0 4.0; 3.0 5.0 6.0]
        @test_throws DomainError statistic(two_1, two_2)
        @test_throws DomainError decision_frechet(HierSample(two_1, 0.0, 6.0), HierSample(two_2, 1.0, 6.0), 0.05, 100, false)

        # both groups have identical rows
        same_1 = repeat([0.5 1.0 2.0], 3)
        same_2 = repeat([0.0 0.25 3.0], 4)
        @test_throws DomainError statistic(same_1, same_2)
        @test isfinite(statistic(same_1, sort(rand(10, 3), dims = 2))) # only one degenerate group

        # observed samples are fine, but with 3 + 3 rows a bootstrap replicate draws one row 3 times in both groups
        # with probability (6 * (1/6)^3)^2 = 1/1296
        h_1 = HierSample([0.0 1.0 2.0; 1.0 3.0 4.0; 2.0 2.0 6.0], 0.0, 6.0)
        h_2 = HierSample([0.0 0.0 1.0; 1.0 2.0 5.0; 3.0 4.0 4.0], 0.0, 5.0)
        @test isfinite(statistic(h_1.atoms, h_2.atoms))
        err = try pvalue_frechet(h_1, h_2, 20000, true); nothing catch e; e end
        @test err isa DomainError
        @test occursin("bootstrap replicate", sprint(showerror, err))
    end

    @testset "Form of p-values and decisions" begin
        h_1, h_2 = test_cases[2]
        n_samples = 200

        p_perm = pvalue_frechet(h_1, h_2, n_samples, false)
        @test 1 / (n_samples + 1) <= p_perm <= 1
        @test isapprox(p_perm * (n_samples + 1), round(p_perm * (n_samples + 1))) # (1 + count) / (n_samples + 1)

        p_boot = pvalue_frechet(h_1, h_2, n_samples, true)
        @test 0 <= p_boot <= 1
        @test isapprox(p_boot * n_samples, round(p_boot * n_samples)) # count / n_samples

        θs = [0.0, 0.01, 0.05, 0.5, 1.0]
        decisions = decision_frechet(h_1, h_2, θs, n_samples, false)
        @test decisions isa Vector{Float64}
        @test all(d -> d == 0.0 || d == 1.0, decisions)
        @test issorted(decisions) # rejection is monotone in θ
        @test decisions[1] == 0.0 && decisions[end] == 1.0
        @test decision_frechet(h_1, h_2, 0.05, n_samples, true) in (0.0, 1.0)
    end

    @testset "Size and power at θ = 0.05" begin
        # Permutation size is exact here since rows are exchangeable under H_0. Bounds are 3 standard errors.
        q = tnormal_normal(0.0, 0.5, -10.0, 10.0)
        q_shift = tnormal_normal(1.0, 0.5, -10.0, 10.0)
        S, θ = 300, 0.05
        rate(q_1, q_2, bootstrap) = mean(decision_frechet(generate_hiersample(q_1, 20, 30), generate_hiersample(q_2, 20, 30),
                                                          θ, 200, bootstrap) for _ in 1:S)
        bound = θ + 3 * sqrt(θ * (1 - θ) / S)

        @test rate(q, q, false) <= bound
        @test rate(q, q, true) <= bound
        @test rate(q, q_shift, false) >= 0.9
        @test rate(q, q_shift, true) >= 0.9
    end
end
