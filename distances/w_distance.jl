# Description: This file contains the functions to compute the Wasserstein distance between two empirical probability measures.

using Statistics
include("../structures.jl")

function wass(p::Vector{Float64}, q::Vector{Float64})
    # computes 1-wasserstein distance between two empirical probability measures with same number of atoms and weights 1/length(atoms)
    return mean(abs.(sort(p) .- sort(q)))
end

function wass(p::emp_pm, q::emp_pm)
    # computes 1-wasserstein distance between two empirical probability measures with same number of atoms and weights 1/length(atoms)
    @assert length(p.atoms) == length(q.atoms)
    return wass(p.atoms, q.atoms)
end
    
function wass(p::emp_pm, q::emp_pm, r::Int)
    # computes r-wasserstein distance between two empirical probability measures with same number of atoms and weights 1/length(atoms)
    @assert length(p.atoms) == length(q.atoms)
    return mean((abs.(sort(p.atoms) .- sort(q.atoms))).^r)^(1/r)
end




