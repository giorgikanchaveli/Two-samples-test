using Test
include("distances/hipm.jl")
include("distributions.jl")
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
    n, m = 150, 200
    atoms_1 = rand(n, m)
    atoms_2 = rand(n, m)
    weights_1 = rand(n, m)
    weights_1 ./= sum(weights_1, dims = 2)
    weights_2 = rand(n, m)
    weights_2 ./= sum(weights_2, dims = 2)

    return atoms_1, weights_1, atoms_2, weights_2
end


function random_h_s(q::LawRPM) 
    n, m = 100, 200
    h_1, h_2 = generate_hiersample(q, n, m), generate_hiersample(q, n, m)
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
    for i in 1:7
        push!(test_cases, random_h_s(DP(i * 1.0 + 1.0, Normal(i*1.0, i*1.2))))
    end
    return test_cases
end

@testset "Checking new HIPM" begin
  

    test_inputs = create_test_cases()

    for x in test_inputs
        old_output = dlip_hugo_marta(to_hugo_marta(x...)...; nRerun=25)[1]
        new_output = dlip(to_mine(x...)...; n_rerun=25)
     
        @test isapprox(old_output, new_output; atol=1e-4, rtol=1e-12)
    end
end







