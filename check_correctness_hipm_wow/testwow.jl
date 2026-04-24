using Test
using Random

include("distances/wow.jl")
include("distributions.jl")
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