include("methods.jl")


# In the simulations we generate hierarchical samples where we associate uniform 
# weight to each random variable. Since weight is determined by the size of the 
# hierarchical sample, we just need to generate atoms (X_{i,j}) for i = 1,...,n and
# j = 1,...,m.

# Simple example for how to use HIPM and WoW.

n = 50
m = 10
hier_sample_1 = rand(n, m)
hier_sample_2 = rand(n, m)

# To compute WoW, we just give as an input hierarchical samples to the function ww.
value_wow = ww(hier_sample_1, hier_sample_2)

# On the other hand, HIPM requires in addition the endpoinds of the interval where random variables
# in the hierarchical samples take values.

a = minimum((minimum(hier_sample_1), minimum(hier_sample_2)))
b = maximum((maximum(hier_sample_1), maximum(hier_sample_2)))
value_dlip = dlip(hier_sample_1, hier_sample_2, a, b)



# In the simulations, I first define the law of the RPM as a struct 
# that is a subtype of the PPM struct. Each such struct includes function
# for generating hierarchical samples. That function returns the struct emp_ppm that 
# encapsulates all information about hierarchical sample — such as atoms, interval endpoints, 
# and the parameters 𝑛 n and 𝑚. 
# Additionally, dlip and WoW are defined for these structs. For convenience, we will continue to refer to any such struct as a hier_sample.

# Example: two Dirichlet processes.

a = 0.0
b = 1.0
q_1 = DP(1.0, Beta(1, 1), a, b)
q_2 = DP(1.0, Beta(2, 3), a, b)

hier_sample_1 = generate_emp(q_1, n, m)
hier_sample_2 = generate_emp(q_2, n, m)


value_wow = ww(hier_sample_1, hier_sample_2)

value_dlip = dlip(hier_sample_1, hier_sample_2, a, b)


# distributions.jl file contains definitions for several laws of RPMs and way to generate
# hierarchical samples from them.



# For estimating the rejection rate, we have function for it in the file methods.jl,
# where we also have functions for energy statistic, wow, hipm, threshold computation, etc.

# Example: estimate rejection rate for all testing schemes (DM, HIPM,WoW, Energy)
# for H_0 : q_1 = q_2  vs H_1: q_1 != q_2 where q_1 and q_2 are Dirichlet processes.


S = 5 # number of times we simulate hierarhical sample to record decisions.
θ = 0.05 # significance level
n_samples = 10 # number of bootstrap/permutation samples
bootstrap = false # boolean variable denoting whether to use bootstrap or permutation approach.

q_1 = DP(10.0, Beta(1,1), 0.0, 1.0) # we set α high because DM method fails if observations are similar.
q_2 = DP(10.0,Beta(1,1), 0.0, 1.0)

n = 5
m = 50
rates_hipm,rates_wow,rates_dm,rates_energy = rejection_rate_all(q_1, q_1, n, m, S, 
    θ, n_samples, bootstrap)


