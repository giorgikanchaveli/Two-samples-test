# optimize wow code


include("distributions.jl")
include("structures.jl")
include("distances/distance_Wasserstein.jl")


using ExactOptimalTransport
using LinearAlgebra

using Tulip



function f_old(q_1::emp_ppm, q_2::emp_ppm, p = 1)
    # Assuming that the number of atoms at the lower level is the same in each measure 
    
    measure1, measure2 = q_1.atoms[:,:], q_2.atoms[:,:]
    
    s1 = size(measure1)
    s2 = size(measure2)

    if s1[2] != s2[2]

        println("PROBLEM OF DIMENSION: each lower measure should have the same dimension")
        return -1. 
    
    else
        
        # timer = TimerOutput()

        # Extract dimensions
        m1 = s1[1]
        m2 = s2[1]
        n = s1[2]

        # Compute matrix of pairwise distances which will be a cost function 

        # @timeit timer "Compute pairwise transports" begin
        
        C = zeros(m1,m2)
        for i=1:m1
            for j =1:m2 
                C[i,j] = wasserstein1DUniform(measure1[i,:],measure2[j,:],p)^p 
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
        output = sum( gamma .* C )
        
        # display(timer)

        return output^(1/p) 

    end

end 



function f_new(q_1::emp_ppm, q_2::emp_ppm, p = 1)
    # Assuming that the number of atoms at the lower level is the same in each measure 
    
    
    
    s1 = size(q_1.atoms)
    s2 = size(q_2.atoms)

    atoms1 = sort(q_1.atoms; dims = 2)
    atoms2 = sort(q_2.atoms; dims = 2)

    if s1[2] != s2[2]

        println("PROBLEM OF DIMENSION: each lower measure should have the same dimension")
        return -1. 
    
    else
        
        # timer = TimerOutput()

        # Extract dimensions
        m1 = s1[1]
        m2 = s2[1]
        n = s1[2]

        # Compute matrix of pairwise distances which will be a cost function 

        # @timeit timer "Compute pairwise transports" begin
        
        C = zeros(m1,m2)
        @inbounds for i=1:m1
            a = atoms1[i,:]
            @inbounds for j =1:m2 
                C[i,j] = wasserstein1DUniform_sorted(a,atoms2[j,:],p)^p 
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
        output = sum( gamma .* C )
        
        # display(timer)

        return output^(1/p) 

    end

end 











# parameters for functions

n = 100
m = 100

a = 0.0
b = 1.0

p = 1
atoms_1 = rand(n,m)
atoms_2 = rand(n,m)

hier_sample_1 = emp_ppm(atoms_1,n,m, a,b)
hier_sample_2 = emp_ppm(atoms_2,n,m, a,b)



# check that values math

old_value = f_old(hier_sample_1,hier_sample_2, p)
new_value = f_new(hier_sample_1,hier_sample_2, p)
@assert abs(old_value - new_value) < 1e-8

# compare times

time_old = @elapsed f_old(hier_sample_1,hier_sample_2, p)
time_new = @elapsed f_new(hier_sample_1,hier_sample_2, p)


println("improvement difference : $(time_old - time_new)")
println("improvement ratio : $(time_old / time_new)")