# This file contains methods to compute the Wasserstein distance for discrete probability measures on R and the 
# wasserstein over wasserstein distance for laws of RPMs.



include("../structures.jl")

using ExactOptimalTransport
using Tulip





"""
    wasserstein_1d_equal
Given two discrete measures with equal number of atoms in R and uniform weights, computes Wasserstein-1 distance. 

# Arguments:
    atoms_1::AbstractVector{Float64}  :  atoms for first probability measure
    atoms_2::AbstractVector{Float64}  :  atoms for second probability measure

# Warning: This function assumes that atoms are sorted.
"""

function wasserstein_1d_equal(atoms_1::AbstractVector{Float64}, atoms_2::AbstractVector{Float64})
    # Atoms must be sorted
    n = length(atoms_1)

    s = 0.0

    @simd for i in 1:n
        s += abs(atoms_1[i] - atoms_2[i])
    end
    s = s / n
    return s
end


"""
    wasserstein_1d_general

Given two discrete measures with possibly different number of atoms in R and uniform weights, computes Wasserstein-1 distance. 

# Arguments:
    atoms_1::AbstractVector{Float64}  :  atoms for first probability measure
    atoms_2::AbstractVector{Float64}  :  atoms for second probability measure

# Warning: This function assumes that atoms are sorted.
"""
function wasserstein_1d_general(atoms_1::AbstractVector{Float64}, atoms_2::AbstractVector{Float64})
    # Atoms must be sorted.

    n_1 = length(atoms_1)
    n_2 = length(atoms_2)

    i = 1
    j = 1
    mass_1 = 1.0 / n_1
    mass_2 = 1.0 / n_2

    cost = 0.0

    while i <= n_1 && j <= n_2
        mass_moved = min(mass_1, mass_2) # mass moved from atoms_1[i] to atoms_2[j]
        cost += mass_moved * abs(atoms_1[i] - atoms_2[j])
        
        mass_1 -= mass_moved # update mass left at atoms_1[i]
        mass_2 -= mass_moved # update mass left at atoms_2[j]

        if abs(mass_1) < 1e-15 # if no mass is left at atoms_1[i], move to atoms_1[i + 1].
            i += 1
            mass_1 = 1.0 / n_1
        end
        if abs(mass_2) < 1e-15 # if no mass is left at atoms_2[j], move to atoms_2[j + 1].
            j += 1
            mass_2 = 1.0 / n_2
        end
    end
    return cost
end


"""
    wasserstein_1d

Given two discrete measures in R with uniform weights accross atoms, computes Wasserstein-1 distance. 

# Arguments:
    atoms_1::AbstractVector{Float64}  :  atoms for first probability measure
    atoms_2::AbstractVector{Float64}  :  atoms for second probability measure

# Warning: This function assumes that atoms are sorted.
"""

function wasserstein_1d(atoms_1::AbstractVector{Float64}, atoms_2::AbstractVector{Float64})
    # Atoms must be sorted
    if length(atoms_1) == length(atoms_2)
        return wasserstein_1d_equal(atoms_1, atoms_2)
    else
        return wasserstein_1d_general(atoms_1, atoms_2)
    end
end





"""
    ww
Computes Wasserstein over Wasserstein distance between two hierarchical samples (definition from paper).

# Arguments:
    atoms_1::AbstractArray{Float64, 2}  :  first hierarchical sample
    atoms_2::AbstractArray{Float64, 2}  :  second hierarchical sample 
"""

function ww(atoms_1::AbstractArray{Float64, 2}, atoms_2::AbstractArray{Float64, 2})
    # elements in each row of the both arrays must be sorted.
    
    size_1 = size(atoms_1)
    size_2 = size(atoms_2)

    # Extract number of rows
    n_1 = size_1[1]
    n_2 = size_2[1]

    # Compute matrix of pairwise distances which will be a cost function 
    
    C = zeros(n_1, n_2)
    for i=1:n_1
        a = view(atoms_1, i, :)
        for j =1:n_2
            C[i,j] = wasserstein_1d(a, view(atoms_2, j, :))
        end
    end
  
    # Build the weights: uniform 
    weight_1 = fill(1 / n_1, n_1)
    weight_2 = fill(1 / n_2, n_2)

    # Solving the optimal transport problem 
    gamma = ExactOptimalTransport.emd(weight_1, weight_2, C, Tulip.Optimizer() )

    #output = sum( gamma .* C )
    return sum(gamma[i,j] * C[i,j] for i in 1:n_1, j in 1:n_2)
end 




"""
    ww
Computes Wasserstein over Wasserstein distance between two hierarchical samples objects.

# Arguments:
    atoms_1::HierSample  :  first hierarchical sample object
    atoms_2::HierSample  :  second hierarchical sample object
"""

function ww(q_1::HierSample, q_2::HierSample)
    # Rows in in both hierarchical samples must be sorted
    return ww(q_1.atoms, q_2.atoms)
end 



















# function ww(q_1::HierSample, q_2::HierSample, p = 1)
#     # Assuming that the number of atoms at the lower level is the same in each measure 
    
#     measure1, measure2 = q_1.atoms[:,:], q_2.atoms[:,:]
    
#     s1 = size(measure1)
#     s2 = size(measure2)

#     if s1[2] != s2[2]

#         println("PROBLEM OF DIMENSION: each lower measure should have the same dimension")
#         return -1. 
    
#     else
        
#         # timer = TimerOutput()

#         # Extract dimensions
#         m1 = s1[1]
#         m2 = s2[1]
#         n = s1[2]

#         # Compute matrix of pairwise distances which will be a cost function 

#         # @timeit timer "Compute pairwise transports" begin
        
#         C = zeros(m1,m2)
#         for i=1:m1
#             for j =1:m2 
#                 C[i,j] = wasserstein1DUniform(measure1[i,:],measure2[j,:],p)^p 
#             end
#         end 

#         # End timer 
#         # end

#         # Build the weights: uniform 
#         weight1 = fill(1 / m1, m1)
#         weight2 = fill(1 / m2, m2)

#         # Solving the optimal transport problem 
#         # @timeit timer "Solve outer OT problem" 
#         gamma = ExactOptimalTransport.emd(weight1, weight2, C, Tulip.Optimizer() )
#         # @timeit timer "compute transport cost" 
#         output = sum( gamma .* C )
        
#         # display(timer)

#         return output^(1/p) 

#     end

# end 

