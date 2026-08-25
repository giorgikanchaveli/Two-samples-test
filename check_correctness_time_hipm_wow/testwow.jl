# Checks that ww (distances/wow.jl) matches Hugo & Marta's original code (hugo_marta_wow.jl).

using Test
using Random
using DataFrames

include("../distances/wow.jl")
include("../distributions.jl")
include("hugo_marta_wow.jl")

function create_test_cases()
    test_cases = Tuple{Matrix{Float64}, Matrix{Float64}}[]

    push!(test_cases, (ones(10, 10), ones(10, 10)))
    for i in 1:7
        push!(test_cases, (rand(10*i, 11*i), rand(10*i, 11*i)))
    end

    return test_cases
end

@testset "Compare old and new WoW implementations" begin
    Random.seed!(1234)

    test_inputs = create_test_cases()

    for (a, b) in test_inputs
        a = sort(a, dims=2)
        b = sort(b, dims = 2)
        old_output = wassersteinOverWasserstein(copy(a), copy(b), 1)
        new_output = ww(copy(a), copy(b))

        @test isapprox(old_output, new_output; atol=1e-5, rtol=1e-12)
    end
end


# Compares average runtime of ww against wassersteinOverWasserstein at fixed sizes n = m = 10, 50, 100,
# averaged over 10 runs each.
#
# Main improvement in ww: it sorts each row once (by the caller) instead of wassersteinOverWasserstein's
# behavior of re-sorting both rows on every single pairwise comparison inside its cost-matrix loop.
# This isn't a large improvement end-to-end though — both implementations spend most of their time
# in the same ExactOptimalTransport.emd call to solve the outer optimal-transport problem, which is
# untouched by the sorting change, so the speedup below is modest (roughly 1-1.4x, not dramatic).

sizes = [10, 50, 100]
n_reps = 10

# Warm-up call so JIT compilation time doesn't skew the first timed run below.
a_warmup, b_warmup = sort(rand(sizes[1], sizes[1]), dims=2), sort(rand(sizes[1], sizes[1]), dims=2)
wassersteinOverWasserstein(copy(a_warmup), copy(b_warmup), 1)
ww(copy(a_warmup), copy(b_warmup))

old_avg_times = Float64[]
new_avg_times = Float64[]

for size in sizes
    a = sort(rand(size, size), dims=2)
    b = sort(rand(size, size), dims=2)

    old_times = Float64[]
    new_times = Float64[]
    for _ in 1:n_reps
        push!(old_times, @elapsed wassersteinOverWasserstein(copy(a), copy(b), 1))
        push!(new_times, @elapsed ww(copy(a), copy(b)))
    end

    push!(old_avg_times, sum(old_times) / n_reps)
    push!(new_avg_times, sum(new_times) / n_reps)
end

times = DataFrame(size = sizes, old_avg_time = old_avg_times, new_avg_time = new_avg_times)
println(times)