include("wow.jl")
using LinearAlgebra 
using BlackBoxOptim
using ExactOptimalTransport
using Tulip

"""
    project_atom

Projects atom x into the closest point on the grid on [a,b]. Grid is equally spaced.

# Arguments:
    x::Float64   :  point to project
    a::Float64   :  left end of interval
    b::Float64   :  right end of interval
    n_grid::Int  :  number of grid points on [a,b].
"""
function project_atom(x::Float64, a::Float64, b::Float64, n_grid::Int)

    x>=a && x <= b || throw(ArgumentError("PROBLEM with the projection: Atom location $x is outside of bounds [$a, $b]"))
    return round(Int,(x - a) / (b-a) * n_grid)+1
end

"""
    project_weights

Projects uniform weights of atoms on the grid on [a,b]. Firstly, we obtain the closest point on the grid from atom and then
associate uniform weight for it.

# Arguments:
    atoms::Vector{Float64}  :  atoms to project
    a::Float64              :  left end of interval
    b::Float64              :  right end of interval
    n_grid::Int             :  number of grid points on [a,b].
"""
function project_weights(atoms::Vector{Float64}, a::Float64, b::Float64, n_grid::Int)
    weights_on_grid = zeros(n_grid+1)

    m = length(atoms)
    for i=1:m 
        weights_on_grid[project_atom(atoms[i],a,b,n_grid)] += 1.0 / m
    end

    return weights_on_grid
end 

"""
    project_weights

Projects weights of atoms on the grid on [a,b]. Firstly, we obtain the closest point on the grid from atom and then
associate its weight for it.

# Arguments:
    atoms::Vector{Float64}    :  atoms to project
    weights::Vector{Float64}  :  weights for each atom
    a::Float64                :  left end of interval
    b::Float64                :  right end of interval
    n_grid::Int               :  number of grid points on [a,b].
"""
function project_weights(atoms::Vector{Float64}, weights::Vector{Float64}, a::Float64, b::Float64, n_grid::Int)
    # We provide specific weights to each atom.
    weights_on_grid = zeros(n_grid+1)

    m = length(atoms)
    for i=1:m 
        weights_on_grid[project_atom(atoms[i],a,b,n_grid)] += weights[i]
    end

    return weights_on_grid
end 



"""
    project_weights_atoms
As we have matrix of atoms, each uniform weights are projected.

# Arguments:
    atoms::AbstractArray{Float64,2}
    a::Float64
    b::Float64
    n_grid::Int
"""
function project_weights_atoms(atoms::AbstractArray{Float64,2},
                a::Float64, b::Float64, n_grid::Int)
    n = size(atoms)[1]
    
    weights_atoms = zeros(n, n_grid + 1)
    for i in 1:n
        weights_atoms[i, :] .= project_weights(atoms[i, :], a, b, n_grid)
    end
    return weights_atoms
end



"""
    project_weights_atoms
As we have matrices of weights and atoms, each weight is projected.

# Arguments:
    atoms::AbstractArray{Float64,2}
    weights::AbstractArray{Float64,2}
    a::Float64
    b::Float64
    n_grid::Int
"""
function project_weights_atoms(atoms::AbstractArray{Float64,2}, weights::AbstractArray{Float64,2},
                a::Float64, b::Float64, n_grid::Int)
    n = size(atoms)[1]
    
    weights_atoms = zeros(n, n_grid + 1)
    for i in 1:n
        weights_atoms[i, :] .= project_weights(atoms[i, :], weights[i,:], a, b, n_grid)
    end
    return weights_atoms
end

function build_eval_matrix_grid(a::Float64, b::Float64, n_grid::Int)

    # Step size 
    delta_x = (b-a) / n_grid

    output = zeros(n_grid+1,n_grid)
    
    # Build via a loop 
    for i=1:n_grid
        output[i+1,1:i] = fill(delta_x,i) 
    end

    return output
end 



function eval_objective_grid(unknown::Vector{Float64}, weights_atoms_1::AbstractArray{Float64, 2},
                weights_atoms_2::AbstractArray{Float64, 2}, Q::Matrix{Float64},derivative::Bool)
    # derivative: true or false, to decide if to output the derivative
    # assumes that number of rows in weights_atoms_1 and weights_atoms_1 are the same.
    
    n = size(weights_atoms_1)[1]
    n_grid = size(Q)[2]

    # Value of the function f on grid point 
    
    f = vcat(0., cumsum(unknown)) * Q[2,1]
    # f = Vector{Float64}(undef, length(unknown) + 1) 
    # f[1] = 0.0
    # cumsum!(@view(f[2:end]), unknown)
    # f = vcat(0., cumsum(unknown)) * Q[2,1]


    # Compute the value of the integral of f 
    integral_f = zeros(2,n)  
    w_1 = view(weights_atoms_1, :, :) # Matrix (n x n_grid)
    mul!(view(integral_f, 1, :), w_1, f)
    w_2 = view(weights_atoms_2, :, :)
    mul!(view(integral_f, 2, :), w_2, f)

    
    permutation_1 = sortperm(integral_f[1,:])
    permutation_2 = sortperm(integral_f[2,:])
    sorted_f_1 = view(integral_f, 1, :)[permutation_1] 
    sorted_f_2 = view(integral_f, 2, :)[permutation_2]

    output = sum(abs.(sorted_f_1 .- sorted_f_2)) * (1.0/n)


    if !derivative 
        return output, 0.
    else
        
        
        # compute derivative: sum_{i = 1}^n s * 1/n * Q' * (weights_atoms_1[permutation_1[i],:] - weights_atoms_2[permutation_2[i],:] )

        output_derivative = zeros(n_grid)
        tempsum = zeros(size(weights_atoms_1)[2])
        for i=1:n 
            
            # Find the sign of the block
            s = sign(integral_f[1,permutation_1[i]] - integral_f[2,permutation_2[i]] )
            
            # Add the right weighted column of the matrix Q, with a sign 
            w1 = view(weights_atoms_1, permutation_1[i], :)
            w2 = view(weights_atoms_2, permutation_2[i], :)
            tempsum .+= s.* (w1 .- w2)
        end 
        # Q' is upper triangular matrice of delta_x with first column equal to 0's. So matrix multiplication is almost reverse of cumsum.
        # When tasted, it didn't make it faster so I don't do it.
        output_derivative = (1.0 / n) .* (Q' * tempsum)
        return output, output_derivative
    end
end


"""
    dlip_projected_measures

Function to compute HIPM after all the weights are projected on the grid.

# Arguments:
    weights_atoms_1::AbstractArray{Float64,2}    
    weights_atoms_2::AbstractArray{Float64,2}  
    b::Float64                                 
    a::Float64                                       
    n_grid::Int = 250                                     
    n_steps::Int=1000                          :  number of steps for Gradient ascent.
    n_rerun::Int = 5                           :  number of times to do optimization algorithm when n_1 = n_2.
    tol::Float64 = 1e-4                        :  tolerance level to stop optimization process when n_1 = n_2.
    max_time::Float64 = 10.0                   :  maximum amount of time to run optimization algorithm when n_1 != n_2.
"""
function dlip_projected_measures(weights_atoms_1::AbstractArray{Float64,2}, weights_atoms_2::AbstractArray{Float64,2},
         a::Float64, b::Float64; 
         n_grid::Int = 250, n_steps::Int=1000, n_rerun::Int = 5,tol::Float64 = 1e-4,
         max_time::Float64 = 10.0)
    size_1 = size(weights_atoms_1)
    size_2 = size(weights_atoms_2)

    n_1 = size_1[1]
    n_2 = size_2[1]

    delta_x = (b-a) / n_grid
    
    if n_1 == n_2
        # weights_atoms_1/2 are of size n_1xn_grid matrix where each row represents the weights on the grid.
           
        # Build the matrix 
        
        Q = build_eval_matrix_grid(a, b, n_grid)

        # Gradient ascent loop 
        # Rerun every time with 5 initial conditions
        # n_rerun = 5
        valueFunction = zeros(n_rerun,n_steps)
        unknownArray = zeros(n_rerun,n_grid)

        for k=1:n_rerun 
    
            # Initial condition 
            # For the first try identity  
            if k==1
                unknown = ones(n_grid)
            else 
                # Random guess via the first few Fourier coefficients
                coeff = 2*rand(3) .- 1.
                unknown = coeff[1]/2 * cos.( LinRange(0,pi,n_grid) ) .+ coeff[2]/4 * cos.( 2 .* LinRange(0,pi,n_grid) ) 
                                    .+ coeff[3]/8 * cos.( 3 .* LinRange(0,pi,n_grid) ) 
                unknown = clamp.(unknown,-1,1)
            end

            # Counter of the steps 
            i = 1
            continueLoop = true 

            while (i <= n_steps) & continueLoop

                # Evaluation of the function and ascent direction 
                f, df = eval_objective_grid(unknown, weights_atoms_1, weights_atoms_2, Q,true)
                
                # Project the ascent direction on the admissible set and find how luch we can advance 
                ascentDirection = df
                t_max = 1. /tol 

                for j =1:n_grid

                    if unknown[j] > 1. - tol
                        unknown[j] = 1.
                        if df[j] > - tol 
                            ascentDirection[j] = 0. 
                        else 
                            t_max = min((-1 - unknown[j]) / ascentDirection[j], t_max)
                        end
                    elseif unknown[j] < -1 + tol
                        unknown[j] = -1.
                        if df[j] < tol 
                            ascentDirection[j] = 0.
                        else 
                            t_max = min((1 - unknown[j]) / ascentDirection[j], t_max)  
                        end
                    else 
                        if ascentDirection[j] > tol 
                            t_max = min((1 - unknown[j]) / ascentDirection[j] ,t_max)
                        elseif ascentDirection[j] < - tol 
                            t_max = min((-1 - unknown[j]) / ascentDirection[j] ,t_max)
                        end  
                                
                    end
                
                end
    
                # Only move if the predicted ascent is large enough in this direction
                if dot(df, ascentDirection) * delta_x  < 1e-8 
                    # Then don't move and exit the loop 
                    candidateUnknown = unknown
                    newf = f
                    continueLoop = false
                else 
                    # Do a backtracking line search 
                    expectedIncrease = dot(df, ascentDirection) 
                    t = t_max 
                    candidateUnknown = unknown + t * ascentDirection 
                    newf = eval_objective_grid(candidateUnknown,weights_atoms_1, weights_atoms_2, Q,false)[1]
                    while (newf < f + 0.5 * t * expectedIncrease) & (t > 1/128) 
                        t *= 0.5 
                        candidateUnknown = unknown + t * ascentDirection
                        newf = eval_objective_grid(candidateUnknown,weights_atoms_1, weights_atoms_2, Q,false)[1]
                    end 

                    # Stop also the loop if the increase is too small 
                    if newf -f <= norm(df) * sqrt(delta_x) * tol^2
                        continueLoop = false 
                    end 

                end

                # Update the variables 
                unknown = candidateUnknown
                valueFunction[k,i] = newf

                # Increase counter 
                i += 1

                # End of loop 
            end

            # At the end of the loop: fill the remaining value with the last one 

            if i < n_steps +1
                valueFunction[k,i:n_steps] = fill( valueFunction[k,i-1], n_steps -i +1 )
            end

            # Store the final function 
            unknownArray[k,:] = unknown
        end
        # Return the maximal value that has been reached, 
        # together with the optimal function
        
        # Best run 
        index = argmax(valueFunction[:,end])
        # Change only return best run
        return valueFunction[index]
        #return maximum(valueFunction[index,end]), Q *unknownArray[index,:] 
    else
        function obj(g::Vector{Float64})
            # we define firstly f from g and then wass distance 
            f = delta_x .* vcat([0.0], cumsum(g))
    
        
            atoms_1 = sort(weights_atoms_1 * f)
            atoms_2 = sort(weights_atoms_2 * f)

            return -1.0 * wasserstein_1d_general(atoms_1, atoms_2)
        end

        res = bboptimize(obj; SearchRange = (-1.0, 1.0), NumDimensions = n_grid, MaxTime = max_time, TraceMode = :silent)
        return -1.0 * best_fitness(res)
    end
end


"""
    dlip

Function to compute HIPM when only atoms are given. 

# Arguments:
    atoms_1::AbstractArray{Float64,2}    
    atoms_2::AbstractArray{Float64,2}  
    b::Float64                                
    a::Float64                                        
    n_grid::Int = 250                                    
    n_steps::Int=1000                          :  number of steps for Gradient ascent.
    n_rerun::Int = 5                           :  number of times to do optimization algorithm when n_1 = n_2.
    tol::Float64 = 1e-4                        :  tolerance level to stop optimization process when n_1 = n_2.
    max_time::Float64 = 10.0                   :  maximum amount of time to run optimization algorithm when n_1 != n_2.
"""
function dlip(atoms_1::AbstractArray{Float64,2}, atoms_2::AbstractArray{Float64,2}, a::Float64, b::Float64; n_grid::Int = 250,
                n_steps::Int=1000, n_rerun::Int = 5,tol::Float64 = 1e-4, max_time::Float64 = 0.5)
    
    # n_1 = size(atoms_1)[1]
    # n_2 = size(atoms_2)[1]
    # # Project weights on a grid
    # weights_atoms_1 = zeros(n_1, n_grid + 1)
    # weights_atoms_2 = zeros(n_2, n_grid + 1)

    # for i in 1:n_1
    #     weights_atoms_1[i, :] .= project_weights(atoms_1[i, :], a, b, n_grid)
    # end

    # for i in 1:n_2
    #     weights_atoms_2[i, :] .= project_weights(atoms_2[i, :], a, b, n_grid)
    # end
    weights_atoms_1 = project_weights_atoms(atoms_1, a, b, n_grid)
    weights_atoms_2 = project_weights_atoms(atoms_2, a, b, n_grid)

    return dlip_projected_measures(weights_atoms_1, weights_atoms_2, a, b; n_grid, n_steps, n_rerun, tol, max_time)
end

"""
    dlip

Function to compute HIPM when only hierarchical sample objects are given. 

# Arguments:
    h_1::HierSample    
    h_2::HierSample  
    b::Float64                                 
    a::Float64                                         
    n_grid::Int = 250                                     
    n_steps::Int=1000                          :  number of steps for Gradient ascent.
    n_rerun::Int = 5                           :  number of times to do optimization algorithm when n_1 = n_2.
    tol::Float64 = 1e-4                        :  tolerance level to stop optimization process when n_1 = n_2.
    max_time::Float64 = 10.0                   :  maximum amount of time to run optimization algorithm when n_1 != n_2.
"""
function dlip(h_1::HierSample, h_2::HierSample, a::Float64, b::Float64; n_grid::Int = 250,
                n_steps::Int=1000, n_rerun::Int = 5,tol::Float64 = 1e-4, max_time::Float64 = 0.5)
    return dlip(h_1.atoms, h_2.atoms, a, b; n_grid, n_steps, n_rerun, tol, max_time)
end



"""
    dlip

Function to compute HIPM when weights are general.

# Arguments:
    atoms_1::AbstractArray{Float64,2}   
    atoms_2::AbstractArray{Float64,2} 
    weights_1::AbstractArray{Float64,2} 
    weights_2::AbstractArray{Float64,2}
    b::Float64                                 
    a::Float64                                         
    n_grid::Int = 250                                     
    n_steps::Int=1000                          :  number of steps for Gradient ascent.
    n_rerun::Int = 5                           :  number of times to do optimization algorithm when n_1 = n_2.
    tol::Float64 = 1e-4                        :  tolerance level to stop optimization process when n_1 = n_2.
    max_time::Float64 = 10.0                   :  maximum amount of time to run optimization algorithm when n_1 != n_2.
"""
function dlip(atoms_1::AbstractArray{Float64,2}, atoms_2::AbstractArray{Float64,2},
              weights_1::AbstractArray{Float64,2}, weights_2::AbstractArray{Float64,2},
              a::Float64, b::Float64; n_grid::Int = 250,
                n_steps::Int=1000, n_rerun::Int = 5,tol::Float64 = 1e-4, max_time::Float64 = 0.5)
    
    weights_atoms_1 = project_weights_atoms(atoms_1, weights_1, a, b, n_grid)
    weights_atoms_2 = project_weights_atoms(atoms_2, weights_2, a, b, n_grid)
    return dlip_projected_measures(weights_atoms_1, weights_atoms_2, a, b; n_grid, n_steps, n_rerun, tol, max_time)
end
