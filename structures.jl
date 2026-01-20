# Every law of a random probability measure(RPM) should be of type LawRPM. Then we define specific ways to generate 
# hierarchical samples from these laws. 

# Every hieraerchical sample should be of type HierSample, which contains the data (hierarchical sample) and the end points of the interval from which atoms are drawn.





"""

    HierSample(atoms, a, b)

Object for a hierarchical sample generated from a law of RPM.
    
# Description: 
        atoms: n x m matrix where each row i is generated from the latent probability measure P_i and P_1,...,P_n ~ law of RPM.
        a: left end of the  interval containing all atoms
        b: right end of the interval containing all atoms.

# Arguments:
        atoms::Matrix{<:Real}
        a::<:Real
        b::<:Real

# Invariants: 
        a <= b
"""
struct HierSample{T <: Real}
    # structure to store data for exchangeable sequences case

    atoms::Matrix{T} # contains all observed random variables
    a::T # interval [a,b] from which atoms are drawn
    b::T # interval [a,b] from which atoms are drawn

    function HierSample(atoms::Matrix{T}, a::T, b::T) where T
        a <= b || throw(ArgumentError("a must be smaller than or equal to b, got a = $a, b = $b."))
    
        new{T}(atoms, a, b)
    end
end


"""
    LawRPM

Type for all laws of Random probability measures.

# Description: 

Every law random probability measure should be a subtype of LawRPM because
some functions will be defined on laws of random probability measure in general instead of on specific law.
"""
abstract type LawRPM end 


