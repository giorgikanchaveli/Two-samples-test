using CSV, DataFrames
using RCall
using Plots
using Distributions, Random

include("distances/hipm.jl")


group1 = ["belarus", "Bulgaria", "Czechia", "Estonia", "Hungary", "Latvia", "Poland", "Lithuania", "Russia", "Slovakia", "Ukraine"]

group2 = ["Australia", "Austria", "Belgium", "Canada", "Denmark", "Finland", "France", "Iceland", "Ireland", "Italy", 
"Japan", "Luxembourg", "Netherlands", "NewZealand", "Norway", "Spain", "Sweden",
"Switzerland", "UnitedKingdom" , "UnitedStatesofAmerica"]

# 111 * (i - 1) + 1, 111*i i denotes the time periods. 


function get_weights(fullpath::String, t::Int, max_age::Int) 
    # t : exact year
    df = open(fullpath) do io
    readline(io)  # ignore metadata line
    CSV.read(io, DataFrame;
             delim=' ',
             ignorerepeated=true)
    end
    # find first index where we start year
    start = findfirst(==(t), df[!,:Year])

    # we truncate age interval to [0, 80], so we have to renormalize death counts.

    column_type = eltype(df[!, "dx"])
    if column_type <: AbstractString 
        dx = parse.(Int, df[start:(start + max_age), "dx"])
    else
        dx = df[start:(start + max_age), "dx"]
    end
    pmf_age_of_death = dx ./ sum(dx)
    return pmf_age_of_death
end

function get_matrix(group::Vector{String},group_number::Int, t::Int, gender::String, max_age::Int)
    # t : exact year

    filepath = "mortality_dataset/group"*string(group_number)*"/$(gender)"
    atoms = Float64.(repeat(collect(0:max_age)', length(group)))
    weights = Matrix{Float64}(undef, length(group), max_age + 1)
    
    for i in 1:length(group)
        fullpath = joinpath(filepath,group[i]*"_"*gender*".txt")
        weights[i,:] .= get_weights(fullpath, t, max_age)
        #push!(hier_sample_1,get_row(fullpath, t))
    end
    return atoms, weights
end

function p_value_dm(atoms_1::Matrix{Float64},atoms_2::Matrix{Float64}, 
                    weights_1::Matrix{Float64},weights_2::Matrix{Float64}, n_bootstrap::Int)
    
    n_1 = size(atoms_1, 1)
    n_2 = size(atoms_2, 1)
     
    @rput atoms_1 atoms_2 weights_1 weights_2 n_1 n_2 n_bootstrap
    R"""
    suppressWarnings({ 

    library(frechet)
    # Build din as a list of density values on the grid
    din_1 <- lapply(1:n_1, function(i) weights_1[i, ])
    din_2 <- lapply(1:n_2, function(i) weights_2[i, ])
    din   <- c(din_1, din_2)

    # Build supin as a list of all atoms
    supin_1 <- lapply(1:n_1, function(i) atoms_1[i, ])
    supin_2 <- lapply(1:n_2, function(i) atoms_2[i, ])
    supin   <- c(supin_1, supin_2)

    group <- c(rep(1, n_1), rep(2, n_2))

    result <- DenANOVA(
    din   = din,
    supin = supin,
    group = group,
    optns = list(boot = TRUE, R = n_bootstrap)
    )


    pvalue = result$pvalBoot # returns bootstrap pvalue
    })
    """
    @rget pvalue  
    
    return pvalue
end


function p_value_hipm(atoms_1::Matrix{Float64},atoms_2::Matrix{Float64}, 
                    weights_1::Matrix{Float64},weights_2::Matrix{Float64}, n_samples::Int, bootstrap::Bool, maxTime::Float64)
    n_1 = size(atoms_1,1)
    n_2 = size(atoms_2,1)
    n = n_1 + n_2
    a = 0.0
    b = maximum(atoms_1[1,:])

    T_observed = dlip_diffsize(atoms_1,atoms_2, weights_1, weights_2, a, b, 250, maxTime)
   
    samples = zeros(n_samples)
    total_weights = vcat(weights_1, weights_2) # collect all rows
    if bootstrap
        for i in 1:n_samples
            indices_1 = sample(1:n, n_1; replace = true)
            indices_2 = sample(1:n, n_2; replace = true)

            new_weights_1 = total_weights[indices_1,:] # first rows indexed by n random indices to the weights_1
            new_weights_2 = total_weights[indices_2,:] # first rows indexed by n random indices to the weights_2

            samples[i] = dlip_diffsize(atoms_1, atoms_2, new_weights_1, new_weights_2, a, b, 250, maxTime)
        end
    else
        for i in 1:n_samples
            random_indices = randperm(n) # indices to distribute rows to new hierarchical meausures

            new_weights_1 = total_weights[random_indices[1:n_1],:] # first rows indexed by n random indices to the atoms_1
            new_weights_2 = total_weights[random_indices[n_1+1:end],:] # first rows indexed by n random indices to the atoms_2
        
            samples[i] = dlip_diffsize(atoms_1, atoms_2, new_weights_1, new_weights_2, a, b, 250, maxTime)
        end
    end
    return mean(samples.>=T_observed)
end






gender = "males"
time_periods = collect(1960:2010)
max_age = 80
n_bootstrap = 100
bootstrap = false
maxTime = 0.5

pvalues_dm = zeros(length(time_periods))
pvalues_hipm = zeros(length(time_periods))

for (i, t) in enumerate(time_periods)
    println("time period: $t")
    atoms_1, weights_1 = get_matrix(group1, 1, t, gender, max_age)
    atoms_2, weights_2 = get_matrix(group2, 2, t, gender, max_age)

    pvalue_dm = p_value_dm(atoms_1, atoms_2, weights_1, weights_2, n_bootstrap)
    pvalue_hipm = p_value_hipm(atoms_1, atoms_2, weights_1, weights_2, n_bootstrap, bootstrap, maxTime)
    
    pvalues_dm[i] = pvalue_dm
    pvalues_hipm[i] = pvalue_hipm
end


all_ticks = minimum(time_periods):5:(maximum(time_periods)+1)

scatterplot = scatter(
    time_periods,
    pvalues_dm,
    xticks = all_ticks,
    xlabel = "Time Periods (Years)",
    ylabel = "P-Value", # Updated label to reflect both series
    ylims = (-0.005,0.3005),
    title = "Scatter Plot of P-Values Over Time",
    label = "DM"
)

# Add the second scatterplot to the existing plot object
scatter!(
    scatterplot, 
    time_periods, # Assuming the x-axis data is the same
    pvalues_hipm, 
    label = "HIPM"
)

# Add the horizontal line to the existing plot object
hline!(scatterplot, [0.05], linestyle = :dash, label = "θ = 0.05")
scatterplot
savefig(scatterplot, gender)

# test


# 111 length
# fullpath = "mortality_dataset/group1/males/belarus_males.txt"

# df = open(fullpath) do io
#     readline(io)  # ignore metadata line
#     CSV.read(io, DataFrame;
#              delim=' ',
#              ignorerepeated=true)
#     end

# t = 1962
# start = findfirst(==(t), df[!,:Year])

#     # we truncate age interval to [0, 80], so we have to renormalize death counts.
# dx = df[start:(start + 80), "dx"]
# # sum(df[1:111,"dx"])




