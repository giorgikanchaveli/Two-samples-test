# Every law of a random probability measure should be of type PPM. Then we can define specific ways to generate 
# hierarchical samples from these laws. 

# Every hieraerchical sample should be of type emp_ppm, which contains the data and some parameters, including number of
# probability measures and number of observations from each probability measure, end points of the interval from which atoms are drawn.




# why mutable? because when I generate hierarchical samples from general 2 laws of RPM, a and b might not coincide.
struct emp_ppm
    # structure to store data for exchangeable sequences case

    atoms::Matrix{Float64} # contains all observed random variables
    n::Int # number of probability measures from which observations are generated
    m::Int # number of observations from each probability measure
    a::Float64 # interval [a,b] from which atoms are drawn
    b::Float64 # interval [a,b] from which atoms are drawn

    function emp_ppm(atoms::Matrix{Float64}, n::Int, m::Int, a::Float64, b::Float64)
        @assert size(atoms, 1) == n "n must equal the number of rows in atoms"
        @assert size(atoms, 2) == m "m must equal the number of columns in atoms"
        new(atoms, n, m, a, b)
    end
end


abstract type PPM end # type for probability measure over space of probability measures
# every law random probability measure should be a subtype of PPM because some functions will be defined on laws of random probability measure in general




struct param_problem_ppm 
    # So far I have not used it
    # structure to define the encapsulate 2 laws of RPM and sizes of exchangeable sequences. Might be useful to compactly
    # input hypothesis problem to functions

    p::PPM # probability measure over probability measures from which we generate observations
    q::PPM # probability measure over probability measures from which we generate observations 
    n::Int # number of probability measures from which observations are generated
    m::Int # number of observations from each probability measure
end


