using Plots
using Distributions
include("data_extractors.jl")


group_1 = ["belarus", "Bulgaria", "Czechia", "Estonia", "Hungary", "Latvia", "Poland", "Lithuania", "Russia", "Slovakia", "Ukraine"]
group_2 = ["Australia", "Austria", "Belgium", "Canada", "Denmark", "Finland", "France", "Iceland", "Ireland", "Italy",
           "Japan", "Luxembourg", "Netherlands", "NewZealand", "Norway", "Spain", "Sweden",
           "Switzerland", "UnitedKingdom", "UnitedStatesofAmerica"]
all_countries = vcat(group_1, group_2)

data_bank = load_mortality_data(all_countries)
females_data = data_bank["females"]
males_data = data_bank["males"]

min_age = 0
max_age = 110


# --- Data truncation analysis ---

"""
    country_with_max_data_lost(gender_data, t, age_truncation)

Returns the fraction of deaths lost and the index of the country that loses the most data
when truncating at `age_truncation` for year `t`.

# Arguments:
    gender_data::Dict{String, DataFrame}
    t::Int                :  Year
    age_truncation::Int
"""
function country_with_max_data_lost(gender_data::Dict{String, DataFrame}, t::Int, age_truncation::Int)
    deaths_counts_all = group_deaths_count(gender_data, all_countries, t, 0, max_age)
    max_data_lost = 0.0
    arg_max = 1
    for i in 1:size(deaths_counts_all, 1)
        data_lost = 1.0 - sum(deaths_counts_all[i, 1:age_truncation+1]) / sum(deaths_counts_all[i, :])
        if data_lost > max_data_lost
            max_data_lost = data_lost
            arg_max = i
        end
    end
    return max_data_lost, arg_max
end

"""
    country_with_max_data_lost(gender_data, time_periods, age_truncation)

Returns a dictionary mapping each year to the worst-case data loss and country index
when truncating at `age_truncation`.

# Arguments:
    gender_data::Dict{String, DataFrame}
    time_periods::Vector{Int}
    age_truncation::Int
"""
function country_with_max_data_lost(gender_data::Dict{String, DataFrame}, time_periods::Vector{Int}, age_truncation::Int)
    data_lost_per_year = Dict{Int, Tuple}()
    for time in time_periods
        data_lost_per_year[time] = country_with_max_data_lost(gender_data, time, age_truncation)
    end
    return data_lost_per_year
end

"""
    save_percentage_data_lost(time_periods, age_truncations)

For each age truncation value, plots and saves the maximum percentage of deaths data lost
across all countries per year (females only).

# Arguments:
    time_periods::Vector{Int}
    age_truncations::Vector{Int}
"""
function save_percentage_data_lost(time_periods::Vector{Int}, age_truncations::Vector{Int})
    for age in age_truncations
        data_lost = country_with_max_data_lost(data_bank["females"], time_periods, age)
        data_lost = [100.0 * data_lost[t][1] for t in time_periods]
        sc = scatter(time_periods, data_lost, xlabel = "Year", ylabel = "% lost", ylims = (0.0, 100.0),
                     legend = false, title = "Max % lost using truncation at $(age)")
        dir = joinpath(pwd(), "mortality", "eda_plots", "datalost_under_truncation")
        mkpath(dir)
        savefig(sc, joinpath(dir, "females_data_lost_$(age).png"))
    end
end


# --- Single country PMF ---

"""
    save_pmf_country(time_periods, country)

Plots and saves the PMF of age at death for a single country across multiple years (males).
Used to visually verify that the PMF peaks well within the age range and truncation is not needed.

# Arguments:
    time_periods::Vector{Int}
    country::String
"""
function save_pmf_country(time_periods::Vector{Int}, country::String)
    gender = "males"
    data_gender = data_bank[gender]
    country_index = findfirst(==(country), all_countries)
    @assert country_index !== nothing "Country $country not found in all_countries."

    ages = collect(min_age:1:max_age)
    pl = scatter(title = "PMF for $(country)", xlabel = "age", ylabel = "pmf",
                 ylims = (0.0, 0.06), legend = false)

    for time in time_periods
        _, pmf_all = group_pmf(data_gender, all_countries, time, min_age, max_age)
        scatter!(pl, ages, pmf_all[country_index, :], label = "$(time)")
    end

    dir = joinpath(pwd(), "mortality", "eda_plots", "single_pmfs")
    mkpath(dir)
    savefig(pl, joinpath(dir, "$(gender)_pmf_$(country)_$(min_age)_$(max_age).png"))
end


# --- PMF plots across time periods ---

"""
    pmfs_for_timeperiods(gender_data, group_1, group_2, time_periods, min_age, age_truncation)

Returns a grid plot where each subplot shows the PMFs of all countries in both groups for a given year.

# Arguments:
    gender_data::Dict{String, DataFrame}
    group_1::Vector{String}
    group_2::Vector{String}
    time_periods::Vector{Int}
    min_age::Int
    age_truncation::Int
"""
function pmfs_for_timeperiods(gender_data::Dict{String, DataFrame}, group_1::Vector{String}, group_2::Vector{String},
                           time_periods::Vector{Int}, min_age::Int, age_truncation::Int)
    n_years = length(time_periods)
    plot_list = []

    for (i, t) in enumerate(time_periods)
        @info "Progress: $i / $(n_years)"
        atoms_1, weights_1 = group_pmf(gender_data, group_1, t, min_age, age_truncation)
        _, weights_2 = group_pmf(gender_data, group_2, t, min_age, age_truncation)
        atoms = atoms_1[1, :]

        p = scatter(title = "Year $t", titlefontsize = 10)
        for j in 1:size(weights_1, 1)
            scatter!(p, atoms, weights_1[j, :], color = "green", lw = 2,
                     label = (j == 1) ? "Sov" : "", alpha = 0.2)
        end
        for j in 1:size(weights_2, 1)
            scatter!(p, atoms, weights_2[j, :], color = "brown", lw = 2,
                     label = (j == 1) ? "NonSov" : "", alpha = 0.2)
        end
        push!(plot_list, p)
    end

    return scatter(plot_list..., layout = (4, 4), size = (1200, 1200),
                   xlabel = "Age", ylabel = "Density", ylims = (0.0, 0.06), alpha = 0.2)
end

"""
    save_pmfs_for_timeperiods(gender_data, gender, group_1, group_2, time_periods, min_age, age_truncation)

Saves the PMF grid plot produced by `pmfs_for_timeperiods` to disk.

# Arguments:
    gender_data::Dict{String, DataFrame}
    gender::String
    group_1::Vector{String}
    group_2::Vector{String}
    time_periods::Vector{Int}
    min_age::Int
    age_truncation::Int
"""
function save_pmfs_for_timeperiods(gender_data::Dict{String, DataFrame}, gender::String,
                           group_1::Vector{String}, group_2::Vector{String},
                           time_periods::Vector{Int}, min_age::Int, age_truncation::Int)
    pl = pmfs_for_timeperiods(gender_data, group_1, group_2, time_periods, min_age, age_truncation)
    dir = joinpath(pwd(), "mortality", "eda_plots", "pmf_frechet_timeperiods")
    mkpath(dir)
    savefig(pl, joinpath(dir, "$(gender)_allpmfs_$(min_age)_$(age_truncation).png"))
end


# --- Quantile plots across time periods ---

"""
    quantiles_for_timeperiods(gender_data, group_1, group_2, time_periods, min_age, age_truncation)

Returns a grid plot where each subplot shows the quantile functions of all countries
in both groups for a given year. Group 1 is shown in red, group 2 in green.

# Arguments:
    gender_data::Dict{String, DataFrame}
    group_1::Vector{String}
    group_2::Vector{String}
    time_periods::Vector{Int}
    min_age::Int
    age_truncation::Int
"""
function quantiles_for_timeperiods(gender_data::Dict{String, DataFrame}, group_1::Vector{String}, group_2::Vector{String},
                           time_periods::Vector{Int}, min_age::Int, age_truncation::Int)
    n_years = length(time_periods)
    plot_list = []
    αs = collect(0.0:0.01:1.0)

    for (i, t) in enumerate(time_periods)
        @info "Progress: $i / $(n_years)"
        atoms_1, weights_1 = group_pmf(gender_data, group_1, t, min_age, age_truncation)
        _, weights_2 = group_pmf(gender_data, group_2, t, min_age, age_truncation)
        atoms = atoms_1[1, :]

        p = plot(title = "Year $t", titlefontsize = 10)
        for j in 1:size(weights_1, 1)
            pmf_obj = DiscreteNonParametric(atoms, weights_1[j, :])
            q = quantile.(pmf_obj, αs)
            plot!(p, q, αs, xlabel = "quantile", ylabel = "Probability level", color = "red", lw = 2,
                  label = (j == 1) ? "Group 1" : "", alpha = 0.5)
        end
        for j in 1:size(weights_2, 1)
            pmf_obj = DiscreteNonParametric(atoms, weights_2[j, :])
            q = quantile.(pmf_obj, αs)
            plot!(p, q, αs, xlabel = "quantile", ylabel = "Probability level", color = "green", lw = 2,
                  label = (j == 1) ? "Group 2" : "", alpha = 0.5)
        end
        push!(plot_list, p)
    end

    return plot(plot_list..., layout = (1, 3), size = (700, 360))
end

"""
    save_quantiles_for_timeperiods(gender_data, gender, group_1, group_2, time_periods, min_age, age_truncation)

Saves the quantile grid plot produced by `quantiles_for_timeperiods` to the quantile_plots directory.

# Arguments:
    gender_data::Dict{String, DataFrame}
    gender::String
    group_1::Vector{String}
    group_2::Vector{String}
    time_periods::Vector{Int}
    min_age::Int
    age_truncation::Int
"""
function save_quantiles_for_timeperiods(gender_data::Dict{String, DataFrame}, gender::String,
                           group_1::Vector{String}, group_2::Vector{String},
                           time_periods::Vector{Int}, min_age::Int, age_truncation::Int)
    pl = quantiles_for_timeperiods(gender_data, group_1, group_2, time_periods, min_age, age_truncation)
    dir = joinpath(pwd(), "mortality", "eda_plots", "quantile_plots")
    mkpath(dir)
    savefig(pl, joinpath(dir, "$(gender)_allquantiles_$(min_age)_$(age_truncation).png"))
end


# --- Fréchet mean quantiles ---

"""
    quantiles_frechet_means(gender_data, group, time, min_age, age_truncation, αs)

Returns the average quantile function across all countries in the group for a given year.
Since the quantile of the Fréchet mean equals the average of the individual quantiles,
this serves as the quantile function of the group's Fréchet mean.

# Arguments:
    gender_data::Dict{String, DataFrame}
    group::Vector{String}
    time::Int
    min_age::Int
    age_truncation::Int
    αs::Vector{Float64}   :  probability levels at which to evaluate the quantile function
"""
function quantiles_frechet_means(gender_data::Dict{String, DataFrame}, group::Vector{String}, time::Int,
                    min_age::Int, age_truncation::Int, αs::Vector{Float64})
    atoms, weights = group_pmf(gender_data, group, time, min_age, age_truncation)
    n = size(atoms, 1)
    atoms = atoms[1, :]
    quantiles = [quantile.(DiscreteNonParametric(atoms, weights[i, :]), αs) for i in 1:n]
    return mean(quantiles)
end

"""
    quantiles_frechet_means_for_timeperiods(gender_data, group_1, group_2, time_periods, min_age, age_truncation)

Returns a grid plot comparing the Fréchet mean quantile functions of the two groups
across the given time periods. Group 1 is shown in red, group 2 in green.

# Arguments:
    gender_data::Dict{String, DataFrame}
    group_1::Vector{String}
    group_2::Vector{String}
    time_periods::Vector{Int}
    min_age::Int
    age_truncation::Int
"""
function quantiles_frechet_means_for_timeperiods(gender_data::Dict{String, DataFrame},
                                    group_1::Vector{String}, group_2::Vector{String},
                                    time_periods::Vector{Int}, min_age::Int, age_truncation::Int)
    plots = []
    αs = collect(0.0:0.01:1.0)
    for time in time_periods
        quantiles_1 = quantiles_frechet_means(gender_data, group_1, time, min_age, age_truncation, αs)
        quantiles_2 = quantiles_frechet_means(gender_data, group_2, time, min_age, age_truncation, αs)
        pl = plot(αs, quantiles_1, xlabel = "prob level", ylabel = "quantile",
                  label = "group 1", color = "red", title = "Year $(time)")
        plot!(pl, αs, quantiles_2, label = "group 2", color = "green")
        push!(plots, pl)
    end
    return plot(plots..., layout = (4, 4), size = (1200, 1200))
end

"""
    save_quantiles_frechet_for_time_periods(gender_data, gender, group_1, group_2, time_periods, min_age, age_truncation)

Saves the Fréchet mean quantile grid plot produced by `quantiles_frechet_means_for_timeperiods` to disk.

# Arguments:
    gender_data::Dict{String, DataFrame}
    gender::String
    group_1::Vector{String}
    group_2::Vector{String}
    time_periods::Vector{Int}
    min_age::Int
    age_truncation::Int
"""
function save_quantiles_frechet_for_time_periods(gender_data::Dict{String, DataFrame}, gender::String,
                group_1::Vector{String}, group_2::Vector{String},
                time_periods::Vector{Int}, min_age::Int, age_truncation::Int)
    pl = quantiles_frechet_means_for_timeperiods(gender_data, group_1, group_2, time_periods, min_age, age_truncation)
    dir = joinpath(pwd(), "mortality", "eda_plots", "pmf_frechet_timeperiods")
    mkpath(dir)
    savefig(pl, joinpath(dir, "$(gender)_frechet_$(min_age)_$(age_truncation).png"))
end


# --- Pooled PMFs ---

"""
    pooled_pmfs(gender_data, group, time, min_age, age_truncation)

Returns the arithmetic mean of the PMFs across all countries in the group for a given year.

# Arguments:
    gender_data::Dict{String, DataFrame}
    group::Vector{String}
    time::Int
    min_age::Int
    age_truncation::Int
"""
function pooled_pmfs(gender_data::Dict{String, DataFrame}, group::Vector{String}, time::Int,
                    min_age::Int, age_truncation::Int)
    _, weights = group_pmf(gender_data, group, time, min_age, age_truncation)
    return vec(mean(weights, dims = 1))
end

"""
    pooled_pmfs_for_timeperiods(gender_data, group_1, group_2, time_periods, min_age, age_truncation)

Returns a grid plot comparing the pooled (mean) PMFs of the two groups across the given time periods.

# Arguments:
    gender_data::Dict{String, DataFrame}
    group_1::Vector{String}
    group_2::Vector{String}
    time_periods::Vector{Int}
    min_age::Int
    age_truncation::Int
"""
function pooled_pmfs_for_timeperiods(gender_data::Dict{String, DataFrame},
                                    group_1::Vector{String}, group_2::Vector{String},
                                    time_periods::Vector{Int}, min_age::Int, age_truncation::Int)
    plots = []
    ages = collect(min_age:1:age_truncation)
    for time in time_periods
        pmf_1 = pooled_pmfs(gender_data, group_1, time, min_age, age_truncation)
        pmf_2 = pooled_pmfs(gender_data, group_2, time, min_age, age_truncation)
        pl = scatter(ages, pmf_1, xlabel = "age", ylabel = "pmf",
                     label = "Sov", color = "green", title = "Year $(time)")
        scatter!(pl, ages, pmf_2, label = "NonSov", color = "brown")
        push!(plots, pl)
    end
    return scatter(plots..., layout = (4, 4), size = (1200, 1200), ylims = (0.0, 0.06))
end

"""
    save_pooled_pmfs_for_time_periods(gender_data, gender, group_1, group_2, time_periods, min_age, age_truncation)

Saves the pooled PMF grid plot produced by `pooled_pmfs_for_timeperiods` to disk.

# Arguments:
    gender_data::Dict{String, DataFrame}
    gender::String
    group_1::Vector{String}
    group_2::Vector{String}
    time_periods::Vector{Int}
    min_age::Int
    age_truncation::Int
"""
function save_pooled_pmfs_for_time_periods(gender_data::Dict{String, DataFrame}, gender::String,
                group_1::Vector{String}, group_2::Vector{String},
                time_periods::Vector{Int}, min_age::Int, age_truncation::Int)
    pl = pooled_pmfs_for_timeperiods(gender_data, group_1, group_2, time_periods, min_age, age_truncation)
    dir = joinpath(pwd(), "mortality", "eda_plots", "pooled_pmfs")
    mkpath(dir)
    savefig(pl, joinpath(dir, "$(gender)_pooled_pmf_$(min_age)_$(age_truncation).png"))
end


# --- Script ---

save_pmf_country([2000], "Denmark")
println("done: one country pmf plots.")

time_periods = [1960, 1992, 2008]
save_quantiles_for_timeperiods(females_data, "females", group_1, group_2, time_periods, min_age, max_age)
save_quantiles_for_timeperiods(males_data, "males", group_1, group_2, time_periods, min_age, max_age)
println("done: quantile plots.")