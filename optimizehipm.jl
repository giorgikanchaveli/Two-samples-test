# This is original code (before I modified) for computing HIPM.

include("distributions.jl")


#include("distances/hipm_original.jl")
include("distances/new_distance.jl")
using LinearAlgebra 
using BenchmarkTools
include("distances/distance_Wasserstein.jl")
# Measures are given as a nTop x nBottom x 2 array
# nTop = n in the article 
# nBottom = m in the article 
# nGrid = M in the article 
# a[:,:,1] -> location of the atom 
# a[:,:,2] -> mass of the atom 


# # -----------------------------------------------------------------------------------
# # Auxilliary functions new distance 
# # -----------------------------------------------------------------------------------

function projectionGrid_new(x,a,b,nGrid)

    if (x <a) | (x > b) 
        error("PROBLEM with the projection: Atom location $x is outside of bounds [$a, $b]")
    else
        return round(Int,(x - a) / (b-a) * nGrid)+1
    end 

end



function projectionGridMeasure_new(atoms::Vector{Float64},a,b,nGrid)

    output = zeros(nGrid+1)

    nBottom = length(atoms)
    for i=1:nBottom 
        output[projectionGrid_new(atoms[i],a,b,nGrid)] += 1.0 / nBottom
    end

    return output
end 


function buildEvaluationMatrixGrid_new(a,b,nGrid)

    # Step size 
    deltaX = (b-a) / nGrid

    output = zeros(nGrid+1,nGrid)
    
    # Build via a loop 
    for i=1:nGrid
        output[i+1,1:i] = fill(deltaX,i) 
    end

    return output
     

end 

function suffix_sum(tempsum::Vector{Float64})
    n = length(tempsum)
    s = sum(tempsum) - tempsum[end]

    output = Vector{Float64}(undef, n - 1)
    
    for i in 1:(n-1)
        output[i] = s
        s = s - tempsum[n - i] 
    end
    return output
end



function evaluationObjectiveGrid_new(unknown::Vector{Float64},weights::AbstractArray{Float64, 3},Q::Matrix{Float64},derivative::Bool)
    # derivative: true or false, to decide if to output the derivative

    nTop = size(weights)[2]
    nGrid = size(Q)[2]

    # Value of the function f on grid point 
    # f = Q * unknown
    # Alternative formula
    #f = vcat(0., cumsum(unknown)) * Q[2,1]

    # Change: avoid new allocations
    f = Vector{Float64}(undef, length(unknown) + 1) 
    f[1] = 0.0
    cumsum!(view(f, 2:(nGrid+1)), unknown) # compute cumsum in place
    f .*= Q[2,1]


    

    # Compute the value of the integral of f 
    integralF = zeros(2,nTop)    
    for k = 1:2
        Wk = view(weights, k, :, :) # Matrix (nTop x nGrid)
        I_k = view(integralF, k, :) # Destination vector (nTop)
        # I_k = Wk * f
        mul!(I_k, Wk, f)           
    end
    #integralF[k,:] = weights[k,:,:] * f

    # Change: Avoid allocations and sorting two times in addition.
    permutation1 = sortperm(integralF[1,:])
    permutation2 = sortperm(integralF[2,:])

    sortedF1 = view(integralF, 1, :)[permutation1] 
    sortedF2 = view(integralF, 2, :)[permutation2]

    output = sum(abs.(sortedF1 .- sortedF2)) * (1.0/nTop)

    #output = sum( abs.( sort(integralF[1,:]) - sort(integralF[2,:] ) )) * 1/nTop

    if !derivative 
        return output, 0.
    else

        # Find the increasing permutation for each 
        
        # First compute the value of the function 
       
        # Then output the derivative 

        # Change: Factor out matrix Q' to avoid matrix multiplication nTop times.
        outputderivative = zeros(nGrid)

        tempsum = zeros(size(weights, 3))
        for i=1:nTop 
            
            # Find the sign of the block
            s = sign(integralF[1,permutation1[i]] - integralF[2,permutation2[i]] )
            # Add the right weighted column of the matrix Q, with a sign 

            
            w1 = view(weights, 1, permutation1[i], :)
            w2 = view(weights, 2, permutation2[i], :)
            @. tempsum .+= s.* (w1 .- w2)
            #outputderivative += s * 1/nTop * Q' * (weights[1,permutation1[i],:] - weights[2,permutation2[i],:] )
        end 
        # Change: Q' is upper triangular matrice of deltaX with first column equal to 0's. So matrix multiplication is almost reverse of cumsum.
        # Actually it doesn't make it faster so I don't do it.
        outputderivative = (1.0 / nTop) .* (Q' * tempsum)
        #outputderivative = (1.0 / nTop) * ((b-a) / nGrid) .* optimized_suffix_sum_original_logic(tempsum)
        


        return output, outputderivative
    
    end


end

# -----------------------------------------------------------------------------------
# Function for the new HIPM distance 
# -----------------------------------------------------------------------------------

function dlip_new(atoms_1::AbstractArray{Float64,2}, atoms_2::AbstractArray{Float64,2}, a::Float64, b::Float64, nGrid::Int = 250,nSteps::Int=1000,
                                            nRerun::Int = 5,tol::Float64 = 1e-4)

    s1 = size(atoms_1)
    s2 = size(atoms_2)

    if s1[1] != s2[1] 

        println("PROBLEM OF DIMENSION: should have measures with the same dimensions")
        return -1. 
    
    else
        
        # Extract dimensions
        nTop = s1[1]
        
        # Time step 
        deltaX = (b-a) / nGrid

        # Project the atoms of each measure on the grid 

        # First coordinate is for measure1/measure2, second for the m index and third is the point of the grid
        weightsAtoms = zeros(2,nTop,nGrid+1)

        for i=1:nTop 
            weightsAtoms[1,i,:] = projectionGridMeasure_new(atoms_1[i, :],a,b,nGrid)
            weightsAtoms[2,i,:] = projectionGridMeasure_new(atoms_2[i, :],a,b,nGrid)
        end
        
        # Build the matrix 
        Q = buildEvaluationMatrixGrid_new(a,b,nGrid)

        # Gradient ascent loop 
        # Rerun every time with 5 initial conditions
        # nRerun = 5
        valueFunction = zeros(nRerun,nSteps)
        unknownArray = zeros(nRerun,nGrid)

        for k=1:nRerun 
            
            # Initial condition 
            # For the first try identity  
            if k==1
                unknown = ones(nGrid)
            else 
                # Random guess via the first few Fourier coefficients
                coeff = 2*rand(3) .- 1.
                unknown = coeff[1]/2 * cos.( LinRange(0,pi,nGrid) ) .+ coeff[2]/4 * cos.( 2 .* LinRange(0,pi,nGrid) ) .+ coeff[3]/8 * cos.( 3 .* LinRange(0,pi,nGrid) ) 
                unknown = clamp.(unknown,-1,1)
            end

            # Counter of the steps 
            i = 1
            continueLoop = true 

            while (i <= nSteps) & continueLoop

                # Evaluation of the function and ascent direction 
                f, df = evaluationObjectiveGrid_new(unknown,weightsAtoms, Q,true)
                
                # Project the ascent direction on the admissible set and find how luch we can advance 
                ascentDirection = df
                t_max = 1. /tol 

                for j =1:nGrid

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
                if dot(df, ascentDirection) * deltaX  < 1e-8 
                    # Then don't move and exit the loop 
                    candidateUnknown = unknown
                    newf = f
                    continueLoop = false
                else 
                    # Do a backtracking line search 
                    expectedIncrease = dot(df, ascentDirection) 
                    t = t_max 
                    candidateUnknown = unknown + t * ascentDirection 
                    newf = evaluationObjectiveGrid_new(candidateUnknown,weightsAtoms, Q,false)[1]
                    while (newf < f + 0.5 * t * expectedIncrease) & (t > 1/128) 
                        t *= 0.5 
                        candidateUnknown = unknown + t * ascentDirection
                        newf = evaluationObjectiveGrid_new(candidateUnknown,weightsAtoms, Q,false)[1]
                    end 

                    # Stop also the loop if the increase is too small 
                    if newf -f <= norm(df) * sqrt(deltaX) * tol^2
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

            if i < nSteps +1
                valueFunction[k,i:nSteps] = fill( valueFunction[k,i-1], nSteps -i +1 )
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


    end

end 


function dlip_new(h_1::emp_ppm, h_2::emp_ppm, a::Float64, b::Float64, nGrid::Int = 250)
    return dlip_new(h_1.atoms, h_2.atoms, a, b, nGrid)
end





n = 100
m = 200

atoms_1, atoms_2 = 3 .* rand(n,m), 3 .* rand(n,m)
a = minimum((minimum(atoms_1), minimum(atoms_2)))
b = maximum((maximum(atoms_1), maximum(atoms_2)))

measure_1 = Array{Float64}(undef, n, m, 2)
measure_2 = Array{Float64}(undef, n, m, 2)

measure_1[:,:,1] = atoms_1
measure_2[:,:,1] = atoms_2

measure_1[:,:,2] = fill(1.0 / m, n, m)
measure_2[:,:,2] = fill(1.0 / m, n, m)

original = @btime dlip(measure_1, measure_2, a, b)
new = @btime dlip_new(atoms_1, atoms_2, a, b)
@assert abs(original[1]  - new) < 1e-5

#wwtime = @btime ww(atoms_1, atoms_2)




# q_1 = tnormal_normal(1.0, 2.0, -10.0, 10.0)
# q_2 = tnormal_normal(2.0, 2.0, -10.0, 10.0)
q_1 = DP(1.0, Beta(1, 1),0.0,1.0)
q_2 = DP(1.0, Beta(1, 1),0.0,1.0)

h_1 = generate_emp(q_1, n, m)
h_2 = generate_emp(q_2, n, m)

# h_1 = emp_ppm(atoms_1, n, m, a, b)
# h_2 = emp_ppm(atoms_2, n, m, a, b)


new = @btime dlip_new(h_1, h_2, 0.0, 1.0)
original = @btime dlip(h_1, h_2, 0.0, 1.0)