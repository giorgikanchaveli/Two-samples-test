include("methods.jl")


# write what is input for dlip and wow with example. 

# In the simulations I define functions to generate hierarchical samples for several laws of 
# RPMs. Those functions generate the struct emp_ppm. This struct encapsulates atoms,
# number of rows, number of columns, maximum value of atoms, minimum value of atoms. 
# dlip and WoW are also defined for such structs.

# For estimating the rejection rate, we have function for it in the file methods.jl,
# where we also have functions for energy statistic, wow, hipm, threshold computation, etc.
