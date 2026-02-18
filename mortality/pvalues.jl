# In this file we do Two sample test for two groups of countries per each time period.

using Plots

include("../methods.jl")
include("data_extractors.jl")



"""
    pvalues_hipm

Given mortality data for specific gender, calculates p values per each time for given groups using HIPM.

# Arguments:
    gender_data::Dict{String, DataFrame}
    group_1::Vector{String}
    group_2::Vector{String}
    time_periods::Vector{Int}
    min_age::Int
    age_truncation::Int
    n_permutations::Int
    max_time::Float64  :  number of seconds to run optimization algorithm for calculating hipm.

"""
function pvalues_hipm(gender_data::Dict{String, DataFrame}, group_1::Vector{String}, group_2::Vector{String},
                 time_periods::Vector{Int}, min_age::Int, age_truncation::Int,
                 n_permutations::Int, max_time::Float64) 
    n_years = length(time_periods) 
    pvalues = Vector{Float64}(undef, n_years) # store p values per each time

    for (i, time) in enumerate(time_periods)
        @info "Progess $i / $(n_years)"
        atoms_1, weights_1 = group_pmf(gender_data, group_1, time, min_age, age_truncation)
        atoms_2, weights_2 = group_pmf(gender_data, group_2, time, min_age, age_truncation)
        pvalues[i] = pvalue_hipm(atoms_1, weights_1, atoms_2, weights_2, n_permutations, max_time)
    end
    return pvalues
end    


"""
    pvalues_averaging

Given mortality data for specific gender, calculates p values per each time for given groups using firstly averaging PMs and
then using W-1 distance.

# Arguments:
    gender_data::Dict{String, DataFrame}
    group_1::Vector{String}
    group_2::Vector{String}
    time_periods::Vector{Int}
    min_age::Int
    age_truncation::Int
    n_permutations::Int
"""
function pvalues_averaging(gender_data::Dict{String, DataFrame}, group_1::Vector{String}, group_2::Vector{String},
                 time_periods::Vector{Int}, min_age::Int, age_truncation::Int,
                 n_permutations::Int) 
    n_years = length(time_periods) 
    pvalues = Vector{Float64}(undef, n_years) # store p values per each time

    for (i, time) in enumerate(time_periods)
        @info "Progess $i / $(n_years)"
        atoms_1, weights_1 = group_pmf(gender_data, group_1, time, min_age, age_truncation)
        atoms_2, weights_2 = group_pmf(gender_data, group_2, time, min_age, age_truncation)
        pvalues[i] = pvalue_averaging(atoms_1, weights_1, atoms_2, weights_2, n_permutations)
    end
    return pvalues
end 






"""
    pvalue_pooling

Given two vectors of observations in R (in this case in {0,1,...,age_truncation}), computes p values using Wasserstein-1 distance
and permutation approach.

# Arguments:
    observations_1::Vector{Float64}
    observations_2::Vector{Float64}
    n_permutations::Int
"""
function pvalue_pooling(observations_1::Vector{Float64}, observations_2::Vector{Float64}, n_permutations::Int)
   
    observed_distance = wasserstein_1d_general(sort(observations_1), sort(observations_2))
    
    # get threshold
    all_observations = vcat(observations_1, observations_2)
    n_1 = length(observations_1)
    n_2 = length(observations_2)
    n_total = n_1 + n_2

    samples = Vector{Float64}(undef, n_permutations)
    for i in 1:n_permutations
        random_indices = randperm(n_total) # indices to distribute rows to new hierarchical meausures
        new_observations_1 = view(all_observations, random_indices[1:n_1])
        new_observations_2 = view(all_observations, random_indices[(n_1+1):n_total])
        samples[i] = wasserstein_1d_general(sort(new_observations_1), sort(new_observations_2))
    end
    p_value = mean(samples .>= observed_distance)
    return p_value
end



"""
    pvalues_pooling

Given mortality data for specific gender, calculates p values per each time for given groups by pooling all deaths observations inside groups
and then using Wasserstein-1 distance.

# Arguments:
    gender_data::Dict{String, DataFrame}
    group_1::Vector{String}
    group_2::Vector{String}
    time_periods::Vector{Int}
    min_age::Int
    age_truncation::Int
    n_permutations::Int
"""
function pvalues_pooling(gender_data::Dict{String, DataFrame}, group_1::Vector{String}, group_2::Vector{String},
                 time_periods::Vector{Int}, min_age::Int, age_truncation::Int,
                 n_permutations::Int) 
    n_years = length(time_periods) 
    pvalues = Vector{Float64}(undef, n_years) # store p values per each time

    for (i, time) in enumerate(time_periods)
        @info "Progess $i / $(n_years)"
        observations_1 = pool_group_deaths_count(group_deaths_count(gender_data, group_1, time, min_age, age_truncation))
        observations_2 = pool_group_deaths_count(group_deaths_count(gender_data, group_2, time, min_age, age_truncation))
        pvalues[i] = pvalue_pooling(observations_1, observations_2, n_permutations)
    end
    return pvalues
end    




function save_pvalues(pvalues::Vector{Float64}, time_periods::Vector{Int},
                 title::String, file_name::String)
    sc = scatter(title = title, xlabel = "time periods", ylabel = "p-value", ylims = (-0.015, 1.1))
    scatter!(sc, time_periods, pvalues, label = "hipm")
    hline!(sc, [0.05], linestyle = :dash, label = "θ = 0.05")
    filepath = joinpath(pwd(), "mortality", "pvalues_plots")
    savefig(sc,filepath*"/$(file_name).png")
end


group_1 = ["belarus", "Bulgaria", "Czechia", "Estonia", "Hungary", "Latvia", "Poland", "Lithuania", "Russia", "Slovakia", "Ukraine"]
group_2 = ["Australia", "Austria", "Belgium", "Canada", "Denmark", "Finland", "France", "Iceland", "Ireland", "Italy", 
"Japan", "Luxembourg", "Netherlands", "NewZealand", "Norway", "Spain", "Sweden",
"Switzerland", "UnitedKingdom" , "UnitedStatesofAmerica"]
all_countries = vcat(group_1, group_2)

data_bank = load_mortality_data(all_countries)
females_data = data_bank["females"]


min_age = 0
max_age = 110

time_periods = collect(1960:1:2010)
n_permutations = 100
max_time = 1.0


# for females
t = time()
pvalues_females_hipm = pvalues_hipm(females_data, group_1, group_2, time_periods, min_age, max_age, n_permutations, max_time)
pvalues_females_averaging = pvalues_averaging(females_data, group_1, group_2, time_periods, min_age, max_age, n_permutations)
pvalues_females_pooling = pvalues_pooling(females_data, group_1, group_2, time_periods, min_age, max_age, n_permutations)
dur = time() - t
sc_females = scatter(title = "P-values, females", xlabel = "time periods", ylabel = "p-value", ylims = (-0.015, 1.1))
scatter!(sc_females, time_periods, pvalues_females_averaging, label = "averaging")
scatter!(sc_females, time_periods, pvalues_females_pooling, label = "pooling")
scatter!(sc_females, time_periods, pvalues_females_hipm, label = "HIPM")
hline!(sc_females, [0.05], linestyle = :dash, label = "θ = 0.05")

# for males
pvalues_males_hipm = pvalues_hipm(males_data, group_1, group_2, time_periods, min_age, max_age, n_permutations, max_time)
pvalues_males_averaging = pvalues_averaging(males_data, group_1, group_2, time_periods, min_age, max_age, n_permutations)
pvalues_males_pooling = pvalues_pooling(males_data, group_1, group_2, time_periods, min_age, max_age, n_permutations)
sc = scatter(title = "P-values, males", xlabel = "time periods", ylabel = "p-value", ylims = (-0.015, 1.1))
scatter!(sc_males, time_periods, pvalues_males_averaging, label = "averaging")
scatter!(sc_males, time_periods, pvalues_males_pooling, label = "pooling")
scatter!(sc_males, time_periods, pvalues_males_hipm, label = "HIPM")
hline!(sc_males, [0.05], linestyle = :dash, label = "θ = 0.05")


# save Plots

filepath = joinpath(pwd(), "mortality", "pvalues_plots")
savefig(sc_females,filepath*"/females_pvalues_poolingaveraginghipm.png")
savefig(sc_females,filepath*"/males_pvalues_poolingaveraginghipm.png")




# males_data = data_bank["males"]


# gender_data = data_bank["females"]



# min_age = 0
# max_age = 110



# time_periods = collect(1960:5:2010)


# n_permutations = 100
# max_time = 4.0


# pvalues_males_newpooling = pvalues_newpooling(gender_data, group_1, group_2, time_periods, min_age, max_age, n_permutations)
# pvalues_males_oldpooling = pvalues_hipm(gender_data, group_1, group_2, time_periods, min_age, max_age, n_permutations, max_time, true)




# sc = scatter(title = "two ways to pool", xlabel = "time periods", ylabel = "p-value", ylims = (-0.015, 1.1))
# scatter!(sc, time_periods, pvalues_males_newpooling, label = "new")
# scatter!(sc, time_periods, pvalues_males_oldpooling, label = "old")
# hline!(sc, [0.05], linestyle = :dash, label = "θ = 0.05")

# males

# pvalues_males_notpooled = pvalues_hipm(males_data, group_1, group_2, time_periods, min_age, max_age, n_permutations, max_time, false)
# pvalues_males_pooled = pvalues_hipm(males_data, group_1, group_2, time_periods, min_age, max_age, n_permutations, max_time, true)
# save_pvalues(pvalues_males_notpooled, time_periods, 
            #  "P-values, males, Not Pooled", "pvalues_males_not_pooled")
# save_pvalues(pvalues_males_pooled, time_periods, 
#              "P-values, males, Pooled", "pvalues_males_pooled")



# females

# pvalues_females_notpooled = pvalues_hipm(females_data, group_1, group_2, time_periods, min_age, max_age, n_permutations, max_time, false)
# pvalues_females_pooled = pvalues_hipm(females_data, group_1, group_2, time_periods, min_age, max_age, n_permutations, max_time, true)
# save_pvalues(pvalues_females_notpooled, time_periods, 
            #  "P-values, females, Not Pooled", "pvalues_females_not_pooled")
# save_pvalues(pvalues_females_pooled, time_periods, 
            #  "P-values, females, Pooled", "pvalues_females_pooled")


