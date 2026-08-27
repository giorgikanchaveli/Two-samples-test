# This file contains the structures for Laws of Random probability measures (RPM) and Hierarchical samples. 


# Every law of a random probability measure (RPM) should be of type LawRPM. 
# This is because we want some methods be defined generally for the type LawRPM. 



# Every hierarchical sample should be of type HierSample, which contains the data (hierarchical sample) and the endpoints of the interval in which atoms are drawn.





"""
    HierSample(atoms, a, b)

struct for a hierarchical sample generated from a law of RPM.
    
# Arguments:
        atoms::Matrix{<:Real}  :  n x m matrix where each row i is generated from the latent probability measure P_i and P_1,...,P_n ~ iid from law of RPM.
        a::<:Real  :  left end of the  interval containing all atoms
        b::<:Real  :  right end of the interval containing all atoms.
"""
struct HierSample{T <: Real}
    
    atoms::Matrix{T} 
    a::T 
    b::T 

    function HierSample(atoms::Matrix{T}, a::T, b::T) where T
        a <= b || throw(ArgumentError("a must be smaller than or equal to b, got a = $a, b = $b."))
    
        new{T}(atoms, a, b)
    end
end


"""
    LawRPM

Type for all laws of Random probability measures.

"""
abstract type LawRPM end 


