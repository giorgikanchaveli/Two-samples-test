# Every law of a random probability measure (RPM) should be of type LawRPM. Then we define specific ways to generate 
# hierarchical samples from these laws. 

# Every hierarchical sample should be of type HierSample, which contains the data (hierarchical sample) and the endpoints of the interval from which atoms are drawn.





"""
    HierSample(atoms, a, b)

struct for a hierarchical sample generated from a law of RPM.
    
# Arguments:
        atoms::Matrix{<:Real}  :  n x m matrix where each row i is generated from the latent probability measure P_i and P_1,...,P_n ~ law of RPM.
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
Every law of a random probability measure should be a subtype of LawRPM because
some functions will be defined on laws of random probability measures in general instead of on specific laws.
"""
abstract type LawRPM end 


