# Code to compute the new HIPM dlip in dimension 1
# 
# The main functions of this file are 
# lowerBound
# dlip

include("../structures.jl")
using LinearAlgebra 

# Measures are given as a nTop x nBottom x 2 array
# nTop = n in the article 
# nBottom = m in the article 
# nGrid = M in the article 
# a[:,:,1] -> location of the atom 
# a[:,:,2] -> mass of the atom 

# -----------------------------------------------------------------------------------
# Lower bound (eq. (1) of the article)
# -----------------------------------------------------------------------------------

function lowerBound(measure1, measure2)

    s1 = size(measure1)
    s2 = size(measure2)

    if s1[1] != s2[1] 

        println("PROBLEM OF DIMENSION: should have measures with the same dimensions")
        return -1. 
    
    else
        
        # Extract dimensions
        nTop = s1[1]

        # Build the two vectors of P(f) with f = Id 
        toTransport1 = vec(sum(measure1[:,:,1] .* measure1[:,:,2], dims=2)) 
        toTransport2 = vec(sum(measure2[:,:,1] .* measure2[:,:,2], dims=2))

        # Then do 1d optimal transport with sorting 
        return 1/nTop * sum(abs.( sort(toTransport1) - sort(toTransport2) ))

    end 

end

# -----------------------------------------------------------------------------------
# Auxilliary functions new distance 
# -----------------------------------------------------------------------------------

function projectionGrid(x::Float64, a::Float64, b::Float64, nGrid::Int)

    if (x <a) | (x > b) 
        println("PROBLEM with the projection: wrong bounds")
        return -1
    else
        return round(Int,(x - a) / (b-a) * nGrid)+1
    end 

end



function projectionGridMeasure(measure,a,b,nGrid)

    output = zeros(nGrid+1)

    nBottom = size(measure)[1]
    for i=1:nBottom 
        output[projectionGrid(measure[i,1],a,b,nGrid)] += measure[i,2]
    end

    return output
end 


function buildEvaluationMatrixGrid(a,b,nGrid)

    # Step size 
    deltaX = (b-a) / nGrid

    output = zeros(nGrid+1,nGrid)
    
    # Build via a loop 
    for i=1:nGrid
        output[i+1,1:i] = fill(deltaX,i) 
    end

    return output
     

end 

function evaluationObjectiveGrid(unknown,weights,Q,derivative)
    # derivative: true or false, to decide if to output the derivative

    nTop = size(weights)[2]
    nGrid = size(Q)[2]

    # Value of the function f on grid point 
    # f = Q * unknown
    # Alternative formula
    f = vcat(0., cumsum(unknown)) * Q[2,1]

    # Compute the value of the integral of f 
    integralF = zeros(2,nTop)
    for k =1:2
        integralF[k,:] = weights[k,:,:] * f
    end

    output = sum( abs.( sort(integralF[1,:]) - sort(integralF[2,:] ) )) * 1/nTop

    if !derivative 
        return output, 0.
    else

        # Find the increasing permutation for each 
        permutation1 = sortperm(integralF[1,:])
        permutation2 = sortperm(integralF[2,:])

        # First compute the value of the function 
       
        # Then output the derivative 
        outputderivative = zeros(nGrid)

        for i=1:nTop 
            
            # Find the sign of the block
            s = sign(integralF[1,permutation1[i]] - integralF[2,permutation2[i]] )
            # Add the right weighted column of the matrix Q, with a sign 
            outputderivative += s * 1/nTop * Q' * (weights[1,permutation1[i],:] - weights[2,permutation2[i],:] )

        end 

        return output, outputderivative
    
    end


end

# -----------------------------------------------------------------------------------
# Function for the new HIPM distance 
# -----------------------------------------------------------------------------------

function dlip(measure1::AbstractArray{Float64, 3}, measure2::AbstractArray{Float64, 3}, a::Float64, b::Float64, nGrid::Int = 250,
                     nSteps::Int=1000,nRerun::Int = 5, tol::Float64 = 1e-4)

    s1 = size(measure1)
    s2 = size(measure2)

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
            weightsAtoms[1,i,:] = projectionGridMeasure(measure1[i,:,:],a,b,nGrid)
            weightsAtoms[2,i,:] = projectionGridMeasure(measure2[i,:,:],a,b,nGrid)
        end

        # Build the matrix 
        Q = buildEvaluationMatrixGrid(a,b,nGrid)

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
                f, df = evaluationObjectiveGrid(unknown,weightsAtoms, Q,true)
                
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
                    newf = evaluationObjectiveGrid(candidateUnknown,weightsAtoms, Q,false)[1]
                    while (newf < f + 0.5 * t * expectedIncrease) & (t > 1/128) 
                        t *= 0.5 
                        candidateUnknown = unknown + t * ascentDirection
                        newf = evaluationObjectiveGrid(candidateUnknown,weightsAtoms, Q,false)[1]
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

        #return maximum(valueFunction[index,end]), Q *unknownArray[index,:] 
        return maximum(valueFunction[index,end])

    end

end 


function dlip(q_1::HierSample, q_2::HierSample, a::Float64, b::Float64; nGrid::Int=150, nSteps::Int=1000, nRerun::Int=5, tol::Float64=1e-4)
    n, m = size(q_1.atoms)
    measure1 = similar(q_1.atoms, eltype(q_1.atoms), n, m, 2)
    measure2 = similar(q_2.atoms, eltype(q_2.atoms), n, m, 2)

    @views begin
        copyto!(measure1[:, :, 1], q_1.atoms)
        copyto!(measure2[:, :, 1], q_2.atoms)
        measure1[:, :, 2] .= 1.0 / m
        measure2[:, :, 2] .= 1.0 / m
    end

    return dlip(measure1, measure2, a, b, nGrid, nSteps, nRerun, tol)
end


function lower_bound(q_1::HierSample, q_2::HierSample)
    a, b = q_1.a, q_1.b
    n, m = size(q_1)
    
    measure1 = similar(q_1.atoms, eltype(q_1.atoms), n, m, 2)
    measure2 = similar(q_2.atoms, eltype(q_2.atoms), n, m, 2)

    @views begin
        copyto!(measure1[:, :, 1], q_1.atoms)
        copyto!(measure2[:, :, 1], q_2.atoms)
        measure1[:, :, 2] .= one(eltype(q_1.atoms)) / m
        measure2[:, :, 2] .= one(eltype(q_2.atoms)) / m
    end
    return lowerBound(measure1, measure2)
end