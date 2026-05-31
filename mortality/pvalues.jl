using Plots
using DelimitedFiles

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
    pvalues = Vector{Float64}(undef, n_years)

    for (i, time) in enumerate(time_periods)
        @info "Progress $i / $(n_years)"
        atoms_1, weights_1 = group_pmf(gender_data, group_1, time, min_age, age_truncation)
        atoms_2, weights_2 = group_pmf(gender_data, group_2, time, min_age, age_truncation)
        pvalues[i] = pvalue_hipm(atoms_1, weights_1, atoms_2, weights_2, n_permutations, max_time)
    end
    return pvalues
end


"""
    pvalues_wow

Given mortality data for specific gender, calculates p values per each time for given groups using WoW.

# Arguments:
    gender_data::Dict{String, DataFrame}
    group_1::Vector{String}
    group_2::Vector{String}
    time_periods::Vector{Int}
    min_age::Int
    age_truncation::Int
    n_permutations::Int
"""
function pvalues_wow(gender_data::Dict{String, DataFrame}, group_1::Vector{String}, group_2::Vector{String},
                 time_periods::Vector{Int}, min_age::Int, age_truncation::Int,
                 n_permutations::Int)
    n_years = length(time_periods)
    pvalues = Vector{Float64}(undef, n_years)

    for (i, time) in enumerate(time_periods)
        @info "Progress $i / $(n_years)"
        deaths_count_1 = group_deaths_count(gender_data, group_1, time, min_age, age_truncation)
        deaths_count_2 = group_deaths_count(gender_data, group_2, time, min_age, age_truncation)
        atoms_1 = hier_sample_from_counts(deaths_count_1)
        atoms_2 = hier_sample_from_counts(deaths_count_2)
        pvalues[i] = pvalue_wow(atoms_1, atoms_2, n_permutations)
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
    pvalues = Vector{Float64}(undef, n_years)

    for (i, time) in enumerate(time_periods)
        @info "Progress $i / $(n_years)"
        atoms_1, weights_1 = group_pmf(gender_data, group_1, time, min_age, age_truncation)
        atoms_2, weights_2 = group_pmf(gender_data, group_2, time, min_age, age_truncation)
        pvalues[i] = pvalue_averaging(atoms_1, weights_1, atoms_2, weights_2, n_permutations)
    end
    return pvalues
end


"""
    pvalue_pooling

Given two vectors of observations, computes p value using Wasserstein-1 distance and permutation approach.

# Arguments:
    observations_1::Vector{Float64}
    observations_2::Vector{Float64}
    n_permutations::Int
"""
function pvalue_pooling(observations_1::Vector{Float64}, observations_2::Vector{Float64}, n_permutations::Int)
    observed_distance = wasserstein_1d_general(sort(observations_1), sort(observations_2))

    all_observations = vcat(observations_1, observations_2)
    n_1 = length(observations_1)
    n_2 = length(observations_2)
    n_total = n_1 + n_2

    samples = Vector{Float64}(undef, n_permutations)
    for i in 1:n_permutations
        random_indices = randperm(n_total)
        new_observations_1 = view(all_observations, random_indices[1:n_1])
        new_observations_2 = view(all_observations, random_indices[(n_1+1):n_total])
        samples[i] = wasserstein_1d_general(sort(new_observations_1), sort(new_observations_2))
    end
    return mean(samples .>= observed_distance)
end


"""
    pvalues_pooling

Given mortality data for specific gender, calculates p values per each time for given groups by pooling all deaths
observations inside groups and then using Wasserstein-1 distance.

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
    pvalues = Vector{Float64}(undef, n_years)

    for (i, time) in enumerate(time_periods)
        @info "Progress $i / $(n_years)"
        observations_1 = pool_group_deaths_count(group_deaths_count(gender_data, group_1, time, min_age, age_truncation))
        observations_2 = pool_group_deaths_count(group_deaths_count(gender_data, group_2, time, min_age, age_truncation))
        pvalues[i] = pvalue_pooling(observations_1, observations_2, n_permutations)
    end
    return pvalues
end


"""
    save_pvalues

Saves p-values and corresponding time periods to a space-delimited text file.

# Arguments:
    pvalues::Vector{Float64}
    time_periods::Vector{Int}
    file_name::String
"""
function save_pvalues(pvalues::Vector{Float64}, time_periods::Vector{Int}, file_name::String)
    filepath = joinpath(pwd(), "values", "mortality_dataset")
    mkpath(filepath)
    filepath = joinpath(filepath, "$(file_name).txt")
    open(filepath, "w") do io
        println(io, "p_value time_period")
        writedlm(io, hcat(pvalues, time_periods), ' ')
    end
    println("Values successfully saved to: $filepath")
end


"""
    load_pvalues

Loads previously saved p-values and time periods from disk.

# Arguments:
    file_name::String
"""
function load_pvalues(file_name::String)
    filepath = joinpath(pwd(), "values", "mortality_dataset", "$(file_name).txt")
    data, _ = readdlm(filepath, ' ', Float64; header=true)
    pvalues = data[:, 1]
    time_periods = Int.(data[:, 2])
    return pvalues, time_periods
end


"""
    main

Computes p-values for all methods (HIPM, WoW, averaging, pooling) for both females and males
over the period 1960–2010 and saves each result.
"""
function main()
    group_1 = ["belarus", "Bulgaria", "Czechia", "Estonia", "Hungary", "Latvia", "Poland", "Lithuania", "Russia", "Slovakia", "Ukraine"]
    group_2 = ["Australia", "Austria", "Belgium", "Canada", "Denmark", "Finland", "France", "Iceland", "Ireland", "Italy",
               "Japan", "Luxembourg", "Netherlands", "NewZealand", "Norway", "Spain", "Sweden",
               "Switzerland", "UnitedKingdom", "UnitedStatesofAmerica"]
    all_countries = vcat(group_1, group_2)

    data_bank = load_mortality_data(all_countries)

    min_age = 0
    max_age = 110
    time_periods = collect(1960:1:2010)
    n_permutations = 100
    max_time = 1.0

    for gender in ["females", "males"]
        @info "Computing p-values for $gender..."
        gender_data = data_bank[gender]

        pv_hipm      = pvalues_hipm(gender_data, group_1, group_2, time_periods, min_age, max_age, n_permutations, max_time)
        pv_wow        = pvalues_wow(gender_data, group_1, group_2, time_periods, min_age, max_age, n_permutations)
        pv_averaging  = pvalues_averaging(gender_data, group_1, group_2, time_periods, min_age, max_age, n_permutations)
        pv_pooling    = pvalues_pooling(gender_data, group_1, group_2, time_periods, min_age, max_age, n_permutations)

        save_pvalues(pv_hipm,     time_periods, "pvalues_$(gender)_hipm")
        save_pvalues(pv_wow,      time_periods, "pvalues_$(gender)_wow")
        save_pvalues(pv_averaging, time_periods, "pvalues_$(gender)_averaging")
        save_pvalues(pv_pooling,  time_periods, "pvalues_$(gender)_pooling")
    end
end

main()