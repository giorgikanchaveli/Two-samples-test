
struct emp_ppm
    # structure to store data for exchangeable case
    atoms::Matrix{Float64}
    n::Int # number of probability measures from which observations are generated
    m::Int # number of observations from each probability measure
    a::Float64 # interval [a,b] from which atoms are drawn
    b::Float64 # interval [a,b] from which atoms are drawn
end


abstract type PPM end # type for probability measure over space of probability measures
# every probability measure should be a subtype of PM/PPM because some functions will be defined on probability measure in general




struct param_problem_ppm # parameters for the problem in exchangeable case
    p::PPM  # probability measure over probability measures from which we generate observations
    q::PPM  
    n::Int # number of probability measures from which observations are generated
    m::Int # number of observations from each probability measure
end

# structures for the approaches to determine the threshold
struct param_emp_ppm
    p_aux::PPM # probability measure from which we construct empirical threshold
    s::Int # number of times to sample distance between two measures
end


struct param_perm
    n_shuffles::Int # number of shuffles
end