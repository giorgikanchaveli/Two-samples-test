# Compute the Wasserstein distance and the wasserstein over wasserstein distance 
# The main function of this code is 
# wassersteinOverWasserstein
include("structures.jl")
include("distances/distance_Wasserstein.jl")
include("distributions.jl")
using ExactOptimalTransport
using Tulip

using BenchmarkTools


# Note: wasserstein1DUniform must accept AbstractVector or SubArray!
function wasserstein1DUniform(atoms1::AbstractVector{Float64}, atoms2::AbstractVector{Float64})
    # ... (the implementation you have, using AbstractVector, is fine) ...
    m1 = length(atoms1)
    m2 = length(atoms2)
    if m1 != m2
        error("wasserstein1DUniform: not the same number of atoms ($(m1) != $(m2))")
    end
    m = m1
    s = 0.0
    # @inbounds for avoiding bounds checks, and @simd for vectorization
    @inbounds @simd for i in 1:m
        s += abs(atoms1[i] - atoms2[i])
    end
    return s / m
end


function ww_new(atoms_1::AbstractArray{Float64,2 }, atoms_2::AbstractArray{Float64,2})
    s1 = size(atoms_1)
    s2 = size(atoms_2)

    if s1[1] != s2[1]
        error("PROBLEM OF DIMENSION: each lower measure should have the same dimension")
    end
    
    n1 = s1[2]
    n2 = s2[2]
    
    C = Matrix{Float64}(undef, n1, n2)

    # 💡 The critical optimization: Use @view and optimized loop order (j then i)
    for j=1:n2
        # Use @view to create non-allocating view for the atoms_2 column
        atom_j_view = @view atoms_2[:, j] 
        
        for i =1:n1 
            atom_i_view = @view atoms_1[:, i]

            # Pass the views to the function 
            C[i,j] = wasserstein1DUniform(atom_i_view, atom_j_view)
        end
    end 
    
    # Rest of the code is clean
    weight1 = fill(1 / n1, n1)
    weight2 = fill(1 / n2, n2)
    
    gamma = ExactOptimalTransport.emd(weight1, weight2, C, Tulip.Optimizer() )
    # Optimal way to compute the final cost
    output = sum(gamma[i,j] * C[i,j] for j in 1:n2, i in 1:n1)
    
    return output 
end





n = 100
m = 1000
atoms_1 = rand(n, m)
atoms_2 = rand(n, m)

atoms_1_t = copy(transpose(atoms_1))
atoms_2_t = copy(transpose(atoms_2))


a = @btime ww(atoms_1, atoms_2)
b = @btime ww_new(atoms_1_t, atoms_2_t)
@assert abs(a - b) < 1e-5