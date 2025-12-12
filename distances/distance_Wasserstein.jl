# Compute the Wasserstein distance and the wasserstein over wasserstein distance 
# The main function of this code is 
# wassersteinOverWasserstein
include("../structures.jl")

using ExactOptimalTransport
using Tulip


# Measures are given as a nTop x nBottom x 2 array
# nTop = n in the article 
# nBottom = m in the article 
# nGrid = M in the article 
# a[:,:,1] -> location of the atom 
# a[:,:,2] -> mass of the atom 


# First simplest 1D Wasserstein distance between measures with the same number of atom and equal measures

function wasserstein1DUniform_sorted(atoms1::AbstractVector{Float64}, atoms2::AbstractVector{Float64})
   # p is the exponent 
   
    if length(atoms1)==length(atoms2)
        n = length(atoms1)

        s = 0.0

        @simd for i in 1:n
            s += abs(atoms1[i] - atoms2[i])
        end
        s = s / n
        return s
    else 
        print("ERROR: not the same number of atoms")
        return -1. 
    end 
end


function ww(atoms_1::AbstractArray{Float64, 2}, atoms_2::AbstractArray{Float64, 2})
    # Assuming that the number of atoms at the lower level is the same in each measure and atoms have sorted rows
    
    s1 = size(atoms_1)
    s2 = size(atoms_2)

    
    if s1[2] != s2[2]

        println("PROBLEM OF DIMENSION: each lower measure should have the same dimension")
        return -1. 
    
    else
        
        # timer = TimerOutput()

        # Extract dimensions
        m1 = s1[1]
        m2 = s2[1]
    
        # Compute matrix of pairwise distances which will be a cost function 

        # @timeit timer "Compute pairwise transports" begin
        
        C = zeros(m1,m2)
        for i=1:m1
            a = view(atoms_1, i, :)
            for j =1:m2 

                C[i,j] = wasserstein1DUniform_sorted(a, view(atoms_2, j, :))
            end
        end 
        # End timer 
        # end
        # Build the weights: uniform 
        weight1 = fill(1 / m1, m1)
        weight2 = fill(1 / m2, m2)

        # Solving the optimal transport problem 
        # @timeit timer "Solve outer OT problem" 
        gamma = ExactOptimalTransport.emd(weight1, weight2, C, Tulip.Optimizer() )
       
        # @timeit timer "compute transport cost" 
        #output = sum( gamma .* C )
        output = sum(gamma[i,j] * C[i,j] for i in 1:m1, j in 1:m2)
        # display(timer)
        return output 
    end
end 






function ww(q_1::emp_ppm, q_2::emp_ppm)
    # Assuming that the number of atoms at the lower level is the same in each measure and atoms are have sorted rows
    return ww(q_1.atoms, q_2.atoms)
end 




# function ww(q_1::emp_ppm, q_2::emp_ppm, p = 1)
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

