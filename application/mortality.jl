using CSV, DataFrames
using RCall
using FLoops
using Plots
using Distributions, Random
using LinearAlgebra
using HypothesisTests

include("../distances/hipm.jl")


group1 = ["belarus", "Bulgaria", "Czechia", "Estonia", "Hungary", "Latvia", "Poland", "Lithuania", "Russia", "Slovakia", "Ukraine"]

group2 = ["Australia", "Austria", "Belgium", "Canada", "Denmark", "Finland", "France", "Iceland", "Ireland", "Italy", 
"Japan", "Luxembourg", "Netherlands", "NewZealand", "Norway", "Spain", "Sweden",
"Switzerland", "UnitedKingdom" , "UnitedStatesofAmerica"]

# 111 * (i - 1) + 1, 111*i i denotes the time periods. 


function country_pmf_per_year(fullpath::String, t::Int, min_age::Int, max_age::Int) 
    # t : exact year
    @assert min_age < max_age "Minimum age must be smaller than maximum age."
    df = open(fullpath) do io
    readline(io)  # ignore metadata line
    CSV.read(io, DataFrame;
             delim=' ',
             ignorerepeated=true)
    end
    # find first index where we start year
    index_base_start = findfirst(==(t), df[!,:Year])
    @assert index_base_start !== nothing "Year $t not found in file."

    index_base_end = index_base_start + 90



    # we truncate age interval to [0, 90], so we have to renormalize death counts.

    column_type = eltype(df[!, "dx"])
    if column_type <: AbstractString 
        dx_base = parse.(Float64, df[index_base_start:(index_base_end), "dx"])
    else
        dx_base = df[index_base_start:(index_base_end), "dx"]
    end
    
    dx_band = dx_base[min_age+1:max_age+1]
    pmf_band = dx_band ./ sum(dx_band)
    return pmf_band
end

function group_pmf_per_year(group::Vector{String},group_number::Int, t::Int, 
                gender::String, min_age::Int, max_age::Int)
    
    # atoms are min_age, min_age + 1,..., max_age.
    
    # t : exact year
    @assert max_age <= 90 "maximum age must be less than or equal to 90."
    @assert min_age >= 0 "minimum age must be higher than or equal to 0."
    @assert min_age < max_age "Minimum age must be smaller than maximum age."
    
    filepath = "application/mortality_dataset/group"*string(group_number)*"/$(gender)"
    
    
    ages = Float64.(collect(min_age:max_age))
    atoms = repeat(ages', length(group), 1) 
    weights = Matrix{Float64}(undef, length(group), length(ages))


    for i in 1:length(group)
        fullpath = joinpath(filepath,group[i]*"_"*gender*".txt")
        weights[i,:] .= country_pmf_per_year(fullpath, t, min_age, max_age)
        #push!(hier_sample_1,get_row(fullpath, t))
    end
    return atoms, weights
end



function group_infant_pmf_per_year(group::Vector{String}, group_number::Int,
     t::Int, gender::String)

    filepath = "application/mortality_dataset/group"*string(group_number)*"/$(gender)"
    
    atoms = Float64.(collect(0:1))
    atoms = repeat(atoms', length(group),1)
    weights = Matrix{Float64}(undef, length(group), 2)

    for i in 1:length(group)
        fullpath = joinpath(filepath,group[i]*"_"*gender*".txt")
        df = open(fullpath) do io
        readline(io)  # ignore metadata line
        CSV.read(io, DataFrame;
                delim=' ',
                ignorerepeated=true)
        end
        start = findfirst(==(t), df[!,:Year])
        @assert start !== nothing "Year $t not found in $(group[i])."

        column_type = eltype(df[!, "dx"])
        if column_type <: AbstractString 
            dx = parse.(Float64, df[start:start+90, :dx])
        else
            dx = df[start:start+90, :dx]
        end
        weights[i, 2] = dx[1] / sum(dx)
        weights[i, 1] = 1 - weights[i, 2]
    end
    return atoms, weights
end




function p_value_hipm(atoms_1::Matrix{Float64},atoms_2::Matrix{Float64}, 
                    weights_1::Matrix{Float64},weights_2::Matrix{Float64}, n_samples::Int, bootstrap::Bool, maxTime::Float64)
    n_1 = size(atoms_1,1)
    n_2 = size(atoms_2,1)
    n = n_1 + n_2

    a = atoms_1[1,1]
    b = atoms_1[1,end]

    T_observed = dlip_diffsize(atoms_1,atoms_2, weights_1, weights_2, a, b, maxTime = maxTime)
   
    samples = zeros(n_samples)
    total_weights = vcat(weights_1, weights_2) # collect all rows
    if bootstrap
        for i in 1:n_samples
            indices_1 = sample(1:n, n_1; replace = true)
            indices_2 = sample(1:n, n_2; replace = true)

            new_weights_1 = total_weights[indices_1,:] # first rows indexed by n random indices to the weights_1
            new_weights_2 = total_weights[indices_2,:] # first rows indexed by n random indices to the weights_2

            samples[i] = dlip_diffsize(atoms_1, atoms_2, new_weights_1, new_weights_2, a, b, maxTime = maxTime)
        end
    else
        for i in 1:n_samples
            random_indices = randperm(n) # indices to distribute rows to new hierarchical meausures

            new_weights_1 = total_weights[random_indices[1:n_1],:] # first rows indexed by n random indices to the atoms_1
            new_weights_2 = total_weights[random_indices[n_1+1:end],:] # first rows indexed by n random indices to the atoms_2
        
            samples[i] = dlip_diffsize(atoms_1, atoms_2, new_weights_1, new_weights_2, a, b, maxTime = maxTime)
        end
    end
    return mean(samples.>=T_observed)
end


function p_value_ks(x::Vector{Float64}, y::Vector{Float64}, 
                    n_samples, bootstrap)

    n_1 = length(x)
    n_2 = length(y)
    n = n_1 + n_2
   
    T_observed = HypothesisTests.ksstats(x, y)[3]
    
    samples = zeros(n_samples)

    all_observations = vcat(x, y) # collect all rows
    if bootstrap
        for i in 1:n_samples
            indices_1 = sample(1:n, n_1; replace = true)
            indices_2 = sample(1:n, n_2; replace = true)

            new_x = all_observations[indices_1] # first rows indexed by n random indices to the weights_1
            new_y = all_observations[indices_2] # first rows indexed by n random indices to the weights_2

            samples[i] = HypothesisTests.ksstats(new_x, new_y)[3]
        end
    else
        for i in 1:n_samples
            random_indices = randperm(n) # indices to distribute rows to new hierarchical meausures

            new_x = all_observations[random_indices[1:n_1]] # first rows indexed by n random indices to the atoms_1
            new_y = all_observations[random_indices[n_1+1:end]] # first rows indexed by n random indices to the atoms_2
        
            samples[i] = HypothesisTests.ksstats(new_x, new_y)[3]
        end
    end
    return mean(samples.>=T_observed)
end





function all_pvalues(time_periods::Vector{Int64}, gender::String, min_age::Int, 
        max_age::Int, n_samples::Int, bootstrap::Bool, maxTime::Float64)
    
    pvalues_ks = zeros(length(time_periods))
    pvalues_hipm = zeros(length(time_periods))

    @floop ThreadedEx() (i, t) in enumerate(time_periods)
        println("time period: $t")
        
        atoms_1, weights_1 = group_pmf_per_year(group1, 1, t, gender, min_age, max_age)
        atoms_2, weights_2 = group_pmf_per_year(group2, 2, t, gender, min_age, max_age)

        #pvalue_dm = p_value_dm_smooth(atoms_1, atoms_2, weights_1, weights_2, n_samples)
        pvalue_hipm = p_value_hipm(atoms_1, atoms_2, weights_1, weights_2, 
                                    n_samples, bootstrap, maxTime)
        pvalue_ks = p_value_ks(weights_1[:,2], weights_2[:,2], n_samples, bootstrap)
       # pvalues_dm[i] = pvalue_dm
        @reduce pvalues_hipm[i] = pvalue_hipm
        @reduce pvalues_ks[i] = pvalue_ks

    end
    #pvalues_dm, pvalues_hipm
    return pvalues_hipm, pvalues_ks
end


function plot_p_values_hipm(pvalues_hipm::Vector{Float64}, time_periods::Vector{Int64}, 
                        gender::String)

    all_ticks = minimum(time_periods):5:(maximum(time_periods)+1)
    ymax = maximum(pvalues_hipm) * 1.1
    scatterplot = scatter(
        time_periods,
        pvalues_hipm,
        xticks = all_ticks,
        xlabel = "Time Periods (Years)",
        ylabel = "P-Value", # Updated label to reflect both series
        ylims = (-0.005,ymax),
        title = "Scatter Plot of P-Values Over Time, $(gender)",
        label = "HIPM"
    )
    # # # Add the horizontal line to the existing plot object
    hline!(scatterplot, [0.05], linestyle = :dash, label = "θ = 0.05")
    return scatterplot
end

# gender = "males"
# time_periods = collect(1960:2010)
# pvalues_hipm = all_pvalues(time_periods, gender, 0, 110, 100, false, 0.5)


# scatterplot_hipm = plot_p_values_hipm(pvalues_hipm,time_periods,gender)
# savefig(scatterplot_hipm, "allages_males.png")

# gender = "females"
# pvalues_hipm = all_pvalues(time_periods, "females", 0, 110, 100, false, 0.5)

# scatterplot_hipm = plot_p_values_hipm(pvalues_hipm,time_periods,gender)
# savefig(scatterplot_hipm, "allages_females.png")



# function obtain_density_R(weights::Matrix{Float64}; 
#                            h::Float64 = 2.0, 
#                            grid_step::Float64 = 0.1)

#     # Number of countries (rows) and number of age bins (columns)
#     n_countries, n_bins = size(weights)

#     # Maximum age (bins correspond to ages 0:max_age)
#     max_age = n_bins - 1

#     # Histogram bin midpoints: 0.5, 1.5, ..., max_age + 0.5
#     x = collect(0:max_age) .+ 0.5

#     # Grid on which the smoothed density will be evaluated (0 to max_age)
#     grid = collect(0:grid_step:max_age)

#     # Transfer Julia objects to the embedded R session
#     @rput weights x grid h

#     # Run R code for local least squares smoothing
#     R"""
#     library(KernSmooth)

#     # trapezoid integration (more accurate than sum(y*dx))
#     trapz <- function(xx, yy) {
#         sum((yy[-1] + yy[-length(yy)]) * diff(xx) / 2)
#       }

#     # Smooth a single country's age-at-death histogram
#     smooth_one <- function(pmf, x, grid, h) {

#       # Local linear least squares smoothing with bandwidth h
#       fit <- locpoly(x = x, y = pmf,
#                      bandwidth = h,
#                      degree = 1,
#                      gridsize = length(grid),
#                      range.x = range(grid))

#       # Enforce non-negativity of the density
#     y <- approx(x = fit$x, y = fit$y, xout = grid, rule = 2)$y
#     y <- pmax(y, 0)


#       # Normalize so the density integrates to 1 over the age range
#       area <- trapz(grid, y)
#         if (is.finite(area) && area > 0) y <- y / area

#         return(y)
#     }

#     # Apply the smoother to each country (row of weights)
#     D <- t(apply(weights, 1, smooth_one, x = x, grid = grid, h = h))
    
#     """

#     # Retrieve the smoothed density matrix from R
#     @rget D

#     # Return the age grid and the corresponding smoothed densities
#     return grid, D
# end


# function p_value_dm_smooth(atoms_1::Matrix{Float64}, atoms_2::Matrix{Float64},
#                     weights_1::Matrix{Float64}, weights_2::Matrix{Float64},
#                     n_bootstrap::Int; h::Float64 = 2.0, grid_step::Float64 = 0.1)

#     grid1, D1 = obtain_density_R(weights_1; h = h, grid_step = grid_step)
#     grid2, D2 = obtain_density_R(weights_2; h = h, grid_step = grid_step)

#     @assert length(grid1) == length(grid2) && all(grid1 .== grid2)
    
#     n_1 = size(D1, 1)
#     n_2 = size(D2, 1)

#     D = vcat(D1, D2)
#     group = vcat(fill(1, n_1), fill(2, n_2))

#     @rput D grid1 group n_bootstrap

#     R"""
    
#     library(frechet)

#     result <- DenANOVA(
#         din   = D,
#         supin = grid1,
#         group = group,
#         optns = list(boot = TRUE, R = n_bootstrap)
#     )

#     pvalue <- result$pvalBoot
    
#     """

#     @rget pvalue
#     return pvalue
# end



# function p_value_dm(atoms_1::Matrix{Float64},atoms_2::Matrix{Float64}, 
#                     weights_1::Matrix{Float64},weights_2::Matrix{Float64}, n_bootstrap::Int)
    
#     n_1 = size(atoms_1, 1)
#     n_2 = size(atoms_2, 1)
     
#     @rput atoms_1 atoms_2 weights_1 weights_2 n_1 n_2 n_bootstrap
#     R"""
#     suppressWarnings({ 

#     library(frechet)
#     # Build din as a list of density values on the grid
#     din_1 <- lapply(1:n_1, function(i) weights_1[i, ])
#     din_2 <- lapply(1:n_2, function(i) weights_2[i, ])
#     din   <- c(din_1, din_2)

#     # Build supin as a list of all atoms
#     supin_1 <- lapply(1:n_1, function(i) atoms_1[i, ])
#     supin_2 <- lapply(1:n_2, function(i) atoms_2[i, ])
#     supin   <- c(supin_1, supin_2)

#     group <- c(rep(1, n_1), rep(2, n_2))

#     result <- DenANOVA(
#     din   = din,
#     supin = supin,
#     group = group,
#     optns = list(boot = TRUE, R = n_bootstrap)
#     )


#     pvalue = result$pvalBoot # returns bootstrap pvalue
#     })
#     """
#     @rget pvalue  
    
#     return pvalue
# end


# function p_value_hipm(atoms_1::Matrix{Float64},atoms_2::Matrix{Float64}, 
#                     weights_1::Matrix{Float64},weights_2::Matrix{Float64}, n_samples::Int, bootstrap::Bool, maxTime::Float64)
#     n_1 = size(atoms_1,1)
#     n_2 = size(atoms_2,1)
#     n = n_1 + n_2
#     a = 0.0
#     b = maximum(atoms_1[1,:])

#     T_observed = dlip_diffsize(atoms_1,atoms_2, weights_1, weights_2, a, b, maxTime = maxTime)
   
#     samples = zeros(n_samples)
#     total_weights = vcat(weights_1, weights_2) # collect all rows
#     if bootstrap
#         for i in 1:n_samples
#             indices_1 = sample(1:n, n_1; replace = true)
#             indices_2 = sample(1:n, n_2; replace = true)

#             new_weights_1 = total_weights[indices_1,:] # first rows indexed by n random indices to the weights_1
#             new_weights_2 = total_weights[indices_2,:] # first rows indexed by n random indices to the weights_2

#             samples[i] = dlip_diffsize(atoms_1, atoms_2, new_weights_1, new_weights_2, a, b, maxTime = maxTime)
#         end
#     else
#         for i in 1:n_samples
#             random_indices = randperm(n) # indices to distribute rows to new hierarchical meausures

#             new_weights_1 = total_weights[random_indices[1:n_1],:] # first rows indexed by n random indices to the atoms_1
#             new_weights_2 = total_weights[random_indices[n_1+1:end],:] # first rows indexed by n random indices to the atoms_2
        
#             samples[i] = dlip_diffsize(atoms_1, atoms_2, new_weights_1, new_weights_2, a, b, maxTime = maxTime)
#         end
#     end
#     return mean(samples.>=T_observed)
# end



# function p_value_ks(x::Vector{Float64}, y::Vector{Float64}, 
#                     n_samples, bootstrap)

#     n_1 = length(x)
#     n_2 = length(y)
#     n = n_1 + n_2
   
#     T_observed = HypothesisTests.ksstats(x, y)[3]
    
#     samples = zeros(n_samples)

#     all_observations = vcat(x, y) # collect all rows
#     if bootstrap
#         for i in 1:n_samples
#             indices_1 = sample(1:n, n_1; replace = true)
#             indices_2 = sample(1:n, n_2; replace = true)

#             new_x = all_observations[indices_1] # first rows indexed by n random indices to the weights_1
#             new_y = all_observations[indices_2] # first rows indexed by n random indices to the weights_2

#             samples[i] = HypothesisTests.ksstats(new_x, new_y)[3]
#         end
#     else
#         for i in 1:n_samples
#             random_indices = randperm(n) # indices to distribute rows to new hierarchical meausures

#             new_x = all_observations[random_indices[1:n_1]] # first rows indexed by n random indices to the atoms_1
#             new_y = all_observations[random_indices[n_1+1:end]] # first rows indexed by n random indices to the atoms_2
        
#             samples[i] = HypothesisTests.ksstats(new_x, new_y)[3]
#         end
#     end
#     return mean(samples.>=T_observed)
# end


# function all_pvalues(time_periods::Vector{Int64}, gender::String, max_age::Int,
#         n_samples::Int, bootstrap::Bool, maxTime::Float64)
    
#     pvalues_dm = zeros(length(time_periods))
#     pvalues_hipm = zeros(length(time_periods))

#     for (i, t) in enumerate(time_periods)
#         println("time period: $t")
        
#         atoms_1, weights_1 = get_matrix(group1, 1, t, gender, max_age)
#         atoms_2, weights_2 = get_matrix(group2, 2, t, gender, max_age)

#         pvalue_dm = p_value_dm_smooth(atoms_1, atoms_2, weights_1, weights_2, n_samples)
#         pvalue_hipm = p_value_hipm(atoms_1, atoms_2, weights_1, weights_2, 
#                                     n_samples, bootstrap, maxTime)
        
#         pvalues_dm[i] = pvalue_dm
#         pvalues_hipm[i] = pvalue_hipm
#     end
#     pvalues_dm, pvalues_hipm
# end


# function all_pvalues_childdeath(time_periods::Vector{Int64}, gender::String, 
#         n_samples::Int, bootstrap::Bool, maxTime::Float64)
    
#     pvalues_dm = zeros(length(time_periods))
#     pvalues_hipm = zeros(length(time_periods))
#     pvalues_ks = zeros(length(time_periods))

#     for (i, t) in enumerate(time_periods)
#         println("time period: $t")
        
#         atoms_1, weights_1 = group_infant_pmf_per_year(group1, 1, t, gender)
#         atoms_2, weights_2 = group_infant_pmf_per_year(group2, 2, t, gender)

#         pvalue_dm = p_value_dm_smooth(atoms_1, atoms_2, weights_1, weights_2, n_samples)
#         pvalue_hipm = p_value_hipm(atoms_1, atoms_2, weights_1, weights_2, 
#                                     n_samples, bootstrap, maxTime)
#         pvalue_ks = p_value_ks(weights_1[:,2], weights_2[:,2], 
#                                     n_samples, bootstrap)
        
#         pvalues_dm[i] = pvalue_dm
#         pvalues_hipm[i] = pvalue_hipm
#         pvalues_ks[i] = pvalue_ks
#     end
#     return pvalues_dm, pvalues_hipm, pvalues_ks
# end


# function plot_p_values(pvalues_dm::Vector{Float64}, pvalues_hipm::Vector{Float64},
#                         pvalues_ks::Vector{Float64}, time_periods::Vector{Int64}, 
#                         gender::String)

#     all_ticks = minimum(time_periods):5:(maximum(time_periods)+1)
#     ymax = maximum((maximum(pvalues_dm), maximum(pvalues_hipm), maximum(pvalues_ks))) + 0.1

#     scatterplot = scatter(
#         time_periods,
#         pvalues_dm,
#         xticks = all_ticks,
#         xlabel = "Time Periods (Years)",
#         ylabel = "P-Value", # Updated label to reflect both series
#         ylims = (-0.005,ymax),
#         title = "Scatter Plot of P-Values Over Time, $(gender)",
#         label = "DM"
#     ) 

#     # # # Add the second scatterplot to the existing plot object
#     scatter!(
#         scatterplot, 
#         time_periods, # Assuming the x-axis data is the same
#         pvalues_hipm, 
#         label = "HIPM"
#     )

#     scatter!(
#         scatterplot, 
#         time_periods, # Assuming the x-axis data is the same
#         pvalues_ks, 
#         label = "KS"
#     )
#     # # # Add the horizontal line to the existing plot object
#     hline!(scatterplot, [0.05], linestyle = :dash, label = "θ = 0.05")
#     return scatterplot
# end

# function all_pvalues_childdeath_not_smoooth(time_periods::Vector{Int64}, gender::String, 
#         n_samples::Int, bootstrap::Bool, maxTime::Float64)
    
#     pvalues_dm = zeros(length(time_periods))
#     pvalues_hipm = zeros(length(time_periods))
#     pvalues_ks = zeros(length(time_periods))

#     for (i, t) in enumerate(time_periods)
#         println("time period: $t")
        
#         atoms_1, weights_1 = group_infant_pmf_per_year(group1, 1, t, gender)
#         atoms_2, weights_2 = group_infant_pmf_per_year(group2, 2, t, gender)

#         pvalue_dm = p_value_dm(atoms_1, atoms_2, weights_1, weights_2, n_samples)
#         pvalue_hipm = p_value_hipm(atoms_1, atoms_2, weights_1, weights_2, 
#                                     n_samples, bootstrap, maxTime)
#         pvalue_ks = p_value_ks(weights_1[:,2], weights_2[:,2], 
#                                     n_samples, bootstrap)
        
#         pvalues_dm[i] = pvalue_dm
#         pvalues_hipm[i] = pvalue_hipm
#         pvalues_ks[i] = pvalue_ks
#     end
#     return pvalues_dm, pvalues_hipm, pvalues_ks
# end



# gender = "males"
# time_periods = collect(1960:2009)
# max_age = 0
# n_samples = 100
# bootstrap = false
# maxTime = 0.5

# t = time()
# pvalues_dm, pvalues_hipm, pvalues_ks = all_pvalues_childdeath(time_periods, gender, n_samples,
#                  bootstrap,maxTime)
# dur = time() - t
# scatterplot = plot_p_values(pvalues_dm, pvalues_hipm, pvalues_ks, time_periods, gender)
# savefig(scatterplot,"$(gender)_child_birth_death_rate.png")



# pvalues_dm, pvalues_hipm, pvalues_ks = all_pvalues_childdeath_not_smoooth(time_periods, gender, n_samples,
#                  bootstrap,maxTime)
#                  scatterplot = plot_p_values(pvalues_dm, pvalues_hipm, pvalues_ks, time_periods, gender)
# savefig(scatterplot,"$(gender)_child_birth_death_rate_not_smooth.png")

# gender = "females"
# pvalues_dm, pvalues_hipm, pvalues_ks = all_pvalues_childdeath_not_smoooth(time_periods, gender, n_samples,
#                  bootstrap,maxTime)
#                  scatterplot = plot_p_values(pvalues_dm, pvalues_hipm, pvalues_ks, time_periods, gender)
# savefig(scatterplot,"$(gender)_child_birth_death_rate_not_smooth.png")

# # # scatterplot
# # savefig(scatterplot,"females_maxage=1.png")

# # test 


# # gender = "females"
# # time_periods = collect(1960:2009)
# # max_age = 1
# # n_bootstrap = 100
# # bootstrap = false
# # maxTime = 0.5

# # pvalues_dm = zeros(length(time_periods))
# # pvalues_hipm = zeros(length(time_periods))

# # for (i, t) in enumerate(time_periods)
# #     println("time period: $t")
# #     atoms_1, weights_1 = get_matrix(group1, 1, t, gender, max_age)
# #     atoms_2, weights_2 = get_matrix(group2, 2, t, gender, max_age)

# #     pvalue_dm = p_value_dm_smooth(atoms_1, atoms_2, weights_1, weights_2, n_bootstrap)
# #     pvalue_hipm = p_value_hipm(atoms_1, atoms_2, weights_1, weights_2, n_bootstrap, bootstrap, maxTime)
    
# #     pvalues_dm[i] = pvalue_dm
# #     pvalues_hipm[i] = pvalue_hipm
# # end


# # all_ticks = minimum(time_periods):5:(maximum(time_periods)+1)
# # ymax = maximum((maximum(pvalues_dm), maximum(pvalues_hipm))) + 0.1

# # scatterplot = scatter(
# #     time_periods,
# #     pvalues_dm,
# #     xticks = all_ticks,
# #     xlabel = "Time Periods (Years)",
# #     ylabel = "P-Value", # Updated label to reflect both series
# #     ylims = (-0.005,ymax),
# #     title = "Scatter Plot of P-Values Over Time",
# #     label = "DM"
# # )

# # # # Add the second scatterplot to the existing plot object
# # scatter!(
# #     scatterplot, 
# #     time_periods, # Assuming the x-axis data is the same
# #     pvalues_hipm, 
# #     label = "HIPM"
# # )

# # # # Add the horizontal line to the existing plot object
# # hline!(scatterplot, [0.05], linestyle = :dash, label = "θ = 0.05")
# # # scatterplot
# # savefig(scatterplot,"females_maxage=1.png")

# # test





# min_age = 0
# max_age = 110
# ages = collect(min_age:max_age)
# t = 1965
# gender = "males"

# h_1 = group_pmf_per_year(group1, 1, t, gender, min_age, max_age)
# h_2 = group_pmf_per_year(group2, 2, t, gender, min_age, max_age)

# ymax = maximum((maximum(h_1[2]),maximum(h_2[2])))

# pmf_plot_1 = plot(xlabel = "ages", ylabel = "mass", title = "PMFs group 1", 
#         xticks = 0:5:110, legend = false, ylims = (-0.001, ymax))
# pmf_plot_2 = plot(xlabel = "ages", ylabel = "mass", title = "PMFs group 2",
#         xticks = 0:5:110, legend = false, ylims = (-0.001, ymax))

# for i in 1:length(group1)
#    # plot!(pmf_plot, ages, h_1[2][i,:], seriestype=:stem, color = "blue")
#     scatter!(pmf_plot_1, ages, h_1[2][i,:])
    
# end

# for i in 1:length(group2)
#    # plot!(pmf_plot, ages, h_1[2][i,:], seriestype=:stem, color = "blue")
#     scatter!(pmf_plot_2, ages, h_2[2][i,:])
    
# end

# plot(pmf_plot_1, pmf_plot_2,
#      layout = (1, 2),
#      size = (1400, 450))



# # 111 length
# # fullpath = "mortality_dataset/group1/males/belarus_males.txt"

# # df = open(fullpath) do io
# #     readline(io)  # ignore metadata line
# #     CSV.read(io, DataFrame;
# #              delim=' ',
# #              ignorerepeated=true)
# #     end

# # t = 1962
# # start = findfirst(==(t), df[!,:Year])

# #     # we truncate age interval to [0, 80], so we have to renormalize death counts.
# # dx = df[start:(start + 80), "dx"]
# # # sum(df[1:111,"dx"])








# # function group_infant_pmf_per_year(group::Vector{String}, group_number::Int,
# #      t::Int, gender::String)

# #     filepath = "mortality_dataset/group"*string(group_number)*"/$(gender)"
# #     atoms = Float64.(repeat(collect(0:1)', length(group)))
# #     weights = Matrix{Float64}(undef, length(group), 2)

# #     for i in 1:length(group)
# #         fullpath = joinpath(filepath,group[i]*"_"*gender*".txt")
# #         df = open(fullpath) do io
# #         readline(io)  # ignore metadata line
# #         CSV.read(io, DataFrame;
# #                 delim=' ',
# #                 ignorerepeated=true)
# #         end
# #         start = findfirst(==(t), df[!,:Year])
# #         dx = df[start, :dx]
# #         lx = df[start, :lx]

# #         dx = dx isa AbstractString ? parse(Float64, dx) : float(dx)
# #         lx = lx isa AbstractString ? parse(Float64, lx) : float(lx)

# #         weights[i, 2] = dx / lx
# #         weights[i, 1] = 1 - weights[i, 2]
# #     end
# #     return atoms, weights
# # end