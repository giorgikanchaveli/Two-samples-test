# Checks that dlip (distances/hipm.jl) matches Hugo & Marta's original code (hugo_marta_hipm.jl).

using Test
using DataFrames
include("../distances/hipm.jl")
include("../distributions.jl")
include("hugo_marta_hipm.jl")



function to_hugo_marta(atoms_1, weights_1, atoms_2, weights_2)
    s = size(atoms_1)
    measure_1 = zeros(s[1], s[2], 2)
    measure_2 = zeros(s[1], s[2], 2)

    measure_1[:,:,1] = atoms_1
    measure_2[:,:,1] = atoms_2

    measure_1[:,:,2] = weights_1
    measure_2[:,:,2] = weights_2

    a = min(minimum(atoms_1), minimum(atoms_2))
    b = max(maximum(atoms_1), maximum(atoms_2))

    return measure_1, measure_2, a, b
end

function to_mine(atoms_1, weights_1, atoms_2, weights_2)
    a = min(minimum(atoms_1), minimum(atoms_2))
    b = max(maximum(atoms_1), maximum(atoms_2)) 
    return atoms_1, atoms_2, weights_1, weights_2, a, b
end



function weights_uniform_diff_atoms()
    atoms_1 = fill(2.0, (10,15))
    atoms_2 = fill(3.0, (10,15))
    weights = fill(1.0/15, (10, 15))
    return atoms_1, weights, atoms_2, weights
end

function weights_random_random_atoms()
    n, m = 50, 200
    atoms_1 = rand(n, m)
    atoms_2 = rand(n, m)
    weights_1 = rand(n, m)
    weights_1 ./= sum(weights_1, dims = 2)
    weights_2 = rand(n, m)
    weights_2 ./= sum(weights_2, dims = 2)

    return atoms_1, weights_1, atoms_2, weights_2
end


function random_h_s(q::LawRPM) 
    n, m = 30, 200
    h_1, h_2 = generate_hiersample(q, n, m), generate_hiersample(q, n, m)
    atoms_1, atoms_2 = h_1.atoms, h_2.atoms
    weights_1 = fill(1.0 / m, (n, m))
    weights_2 = fill(1.0 / m, (n, m))
    return atoms_1, weights_1, atoms_2, weights_2
end

function random_h_s(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int) 
    
    h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
    atoms_1, atoms_2 = h_1.atoms, h_2.atoms
    weights_1 = fill(1.0 / m, (n, m))
    weights_2 = fill(1.0 / m, (n, m))
    return atoms_1, weights_1, atoms_2, weights_2
end


function create_test_cases()
    test_cases = []

    push!(test_cases, weights_random_random_atoms())
    push!(test_cases, weights_uniform_diff_atoms())
    push!(test_cases, random_h_s(DP(1.0, Uniform(0.0,1.0))))
    push!(test_cases, random_h_s(normal_normal_B(1.0,2.0,3.0)))
    for i in 1:20
        # push!(test_cases, random_h_s(normal_normal_B(1.0,2.0,1.5), normal_normal_A(1.5,1.0)))
        push!(test_cases, random_h_s(DP(1.0, Uniform(0.0,1.0)), DP(2.0, Uniform(0.0,5.0)), 10, 20))
    end
    return test_cases
end

@testset "Checking new HIPM" begin
    test_inputs = create_test_cases()

    for x in test_inputs

        Random.seed!(12345)
        old_output = dlip_hugo_marta(to_hugo_marta(x...)...; nRerun=10, nGrid=500)[1] # nGrid must match n_grid below, or the two aren't comparable
        Random.seed!(12345)
        new_output = dlip(to_mine(x...)...; n_rerun=10, n_grid=500)

        @test isapprox(old_output, new_output; atol=1e-9, rtol=1e-12)
    end
end


# Compares average runtime of dlip against dlip_hugo_marta at fixed sizes n = m = 10, 50, 100,
# averaged over 10 runs each.
#
# Main improvements in dlip's eval_objective_grid (distances/hipm.jl), vs. evaluationObjectiveGrid
# in hugo_marta_hipm.jl:
#   1) Allocations: the old code indexes a 3D array (weights[k,:,:]) and computes weights[k,:,:] * f,
#      each of which allocates a new array on every gradient-ascent step. dlip uses views + mul!
#      (writing directly into a preallocated array) instead.
#   2) The derivative computation in the old code multiplies by Q' (an n_grid x n_grid matrix) inside
#      the per-i loop, i.e. n times per step — an O(n * n_grid^2) cost. dlip accumulates the
#      (cheap, O(n_grid)) weighted sum across i first, then multiplies by Q' once outside the loop
#      (linearity of matrix multiplication), reducing that step to O(n * n_grid + n_grid^2). This is
#      the dominant win, especially at n_grid=500.

sizes = [10, 50, 100]
n_reps = 10
q_1, q_2 = DP(1.0, Uniform(0.0, 1.0)), DP(2.0, Uniform(0.0, 5.0))

# Warm-up call so JIT compilation time doesn't skew the first timed run below.
x_warmup = random_h_s(q_1, q_2, sizes[1], sizes[1])
dlip_hugo_marta(to_hugo_marta(x_warmup...)...; nRerun=10, nGrid=500)
dlip(to_mine(x_warmup...)...; n_rerun=10, n_grid=500)

old_avg_times = Float64[]
new_avg_times = Float64[]

for size in sizes
    x = random_h_s(q_1, q_2, size, size)

    old_times = Float64[]
    new_times = Float64[]
    for _ in 1:n_reps
        push!(old_times, @elapsed dlip_hugo_marta(to_hugo_marta(x...)...; nRerun=10, nGrid=500))
        push!(new_times, @elapsed dlip(to_mine(x...)...; n_rerun=10, n_grid=500))
    end

    push!(old_avg_times, sum(old_times) / n_reps)
    push!(new_avg_times, sum(new_times) / n_reps)
end

times = DataFrame(size = sizes, old_avg_time = old_avg_times, new_avg_time = new_avg_times)
println(times)







