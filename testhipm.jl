using Random
using Distributions
using Test

include("methods.jl")
include("hugo_hipm.jl")

# ----------------------------
# Helpers
# ----------------------------

function hiersample_to_measure(h)
    nTop, nBottom = size(h.atoms)
    measure = zeros(nTop, nBottom, 2)
    measure[:, :, 1] .= h.atoms
    measure[:, :, 2] .= 1.0 / nBottom
    return measure
end


function compare_on_hiersamples(h1, h2; n_grid=250, seed=2024,
                                n_steps=1000, n_rerun=5, tol=1e-4, verbose=true)

    m1 = hiersample_to_measure(h1)
    m2 = hiersample_to_measure(h2)
    a = minimum((h1.a, h2.a))
    b = maximum((h1.b, h2.b))

    Random.seed!(seed)
    v_mine = dlip(h1, h2, a, b; n_grid=n_grid, n_steps=n_steps, n_rerun=n_rerun, tol=tol)

    Random.seed!(seed)
    v_hugo = dlip_hugo(m1, m2, a, b, n_grid, n_steps, n_rerun, tol)[1]

    diff = abs(v_mine - v_hugo)

    if verbose
        println("mine  = ", v_mine)
        println("hugo  = ", v_hugo)
        println("diff  = ", diff)
        println("close = ", isapprox(v_mine, v_hugo; atol=1e-4, rtol=1e-4))
    end

    return v_mine, v_hugo, diff
end

Random.seed!(42)
q1 = normal_normal_A(1.0,2.0)
q2 = normal_normal_B(1.0,1.0,9.0)

for (n, m) in [(1,1), (10,2), (20,3),(90,90)]
    h1 = generate_hiersample(q1, n, m)
    h2 = generate_hiersample(q2, n, m)

    println("\nCase n = $n, m = $m")
    compare_on_hiersamples(h1, h2; n_grid=250, seed=2024, n_steps=300, n_rerun=5, tol=1e-4, verbose=true)
end

println("\nAll tests completed.")