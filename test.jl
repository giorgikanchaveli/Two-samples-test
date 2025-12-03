# optimize wow code

using Distributions, Random
include("distributions.jl")
include("structures.jl")
include("distances/distance_Wasserstein.jl")
include("distances/new_distance.jl")



function wasserstein1DUniform_sorted_new(atoms1::AbstractVector{Float64}, atoms2::AbstractVector{Float64}, p::Int)
   # atoms1 and atoms2 only list of position 
   # p is the exponent 
   
    if length(atoms1)==length(atoms2)
        n = length(atoms1)

        s = 0.0

        @simd for i in 1:n
            s += abs(atoms1[i] - atoms2[i])^p
        end
        s = (s / n)^(1/p)
        return s
    else 
        print("ERROR: not the same number of atoms")
        return -1. 
    end 
end


function ww_new(atoms_1::AbstractArray{Float64, 2}, atoms_2::AbstractArray{Float64, 2}, p = 1)
    # Assuming that the number of atoms at the lower level is the same in each measure 
    
    
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

                C[i,j] = wasserstein1DUniform_sorted_new(a, view(atoms_2, j, :),p)^p 
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
m = 200

# a = 0.0
# b = 1.0


q_1 = tnormal_normal(1.0,1.0,-10.0,10.0)
q_2 = tnormal_normal(1.0,1.0,-10.0,10.0)
q_3 = mixture_ppm(q_1, q_2, 0.5)


h_1,h_2 = generate_emp(q_3,n,m),  generate_emp(q_3,n,m)
t_1 = @time [old_value = ww(h_1,h_2) for i in 1:10]
t_2 = @time [ww_new(h_1.atoms,h_2.atoms) for i in 1:10]

@assert abs(old_value - new_value) <= 1e-5

# check that values math
Random.seed!(12345)
old_value = f_old(q_1, q_2, n, m, S, θ, n_boostrap)
Random.seed!(12345)
new_value = f_new(q_1, q_2, n, m, S, θ, n_boostrap)
@assert abs(old_value - new_value) < 1e-8

# compare times
Random.seed!(12345)
time_old = @elapsed f_old(q_1, q_2, n, m, S, θ, n_boostrap)
Random.seed!(12345)
time_new = @elapsed f_new(q_1, q_2, n, m, S, θ, n_boostrap)


println("improvement difference : $(time_old - time_new)")
println("improvement ratio : $(time_old / time_new)")
println("new time : $(time_new)")


