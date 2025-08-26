struct emp_pm # the data will have such form, it will be input for distance function
    # structure to store data for i.i.d case
    atoms::Vector{Float64}
    a::Float64 # interval [a,b] from which atoms are drawn
    b::Float64 
end


struct emp_ppm
    # structure to store data for exchangeable case
    atoms::Matrix{Float64}
    n::Int # number of probability measures from which observations are generated
    m::Int # number of observations from each probability measure
    a::Float64 # interval [a,b] from which atoms are drawn
    b::Float64 # interval [a,b] from which atoms are drawn
end

abstract type PM end # type for probability measure over space of observations
abstract type PPM end # type for probability measure over space of probability measures
# every probability measure should be a subtype of PM/PPM because some functions will be defined on probability measure in general


# sturcutes for the rejection rate problem
struct param_problem_pm # parameters for the problem in i.i.d case
    p::PM # probability measure from which we generate observations
    q::PM 
    n_rv::Int # number of random variables
end


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

struct param_emp
    p_0::PM # Probability measure from which we construct the empirical threshold
    s::Int # number of times to sample distance between two measures
end

struct param_perm
    n_shuffles::Int # number of shuffles
end