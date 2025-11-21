# optimize wow code


include("distributions.jl")
include("structures.jl")

using ExactOptimalTransport
using Tulip



function wasserstein1DUniform_old(atoms1, atoms2,p)
   # atoms1 and atoms2 only list of position 
   # p is the exponent 
   
    if length(atoms1)==length(atoms2)
        return sum( 1/length(atoms1) * (abs.(sort(atoms1) - sort(atoms2))).^p )^(1/p)
    else 
        print("ERROR: not the same number of atoms")
        return -1. 
    end 
end

function wasserstein1DUniform_new(atoms1::Vector{Float64}, atoms2::Vector{Float64},p::Int)
   # atoms1 and atoms2 only list of position 
   # p is the exponent 
   
    if length(atoms1)==length(atoms2)
        n = length(atoms1)
        a = sort!(copy(atoms1))
        b = sort!(copy(atoms2))

        s = 0.0

        @inbounds for i in 1:n
            s += abs(a[i] - b[i])^p
        end
        s = (s / n)^(1/p)
        return s
    else 
        print("ERROR: not the same number of atoms")
        return -1. 
    end 
end
















n = 1000
p = 10
atoms_1 = rand(n)
atoms_2 = rand(n)


old_value = wasserstein1DUniform_old(atoms_1, atoms_2, p)
new_value = wasserstein1DUniform_new(atoms_1, atoms_2, p)
@assert abs(old_value - new_value) < 1e-8


time_old = @elapsed wasserstein1DUniform_old(atoms_1, atoms_2, p)
time_new = @elapsed wasserstein1DUniform_new(atoms_1, atoms_2, p)


println("improvement difference : $(time_old - time_new)")
println("improvement ratio : $(time_old / time_new)")