using Plots
using Distributions
include("data_extractors.jl")



group_1 = ["belarus", "Bulgaria", "Czechia", "Estonia", "Hungary", "Latvia", "Poland", "Lithuania", "Russia", "Slovakia", "Ukraine"]

group_2 = ["Australia", "Austria", "Belgium", "Canada", "Denmark", "Finland", "France", "Iceland", "Ireland", "Italy", 
"Japan", "Luxembourg", "Netherlands", "NewZealand", "Norway", "Spain", "Sweden",
"Switzerland", "UnitedKingdom" , "UnitedStatesofAmerica"]

all_countries = vcat(group_1, group_2)


data_bank = load_mortality_data(["males", "females"], all_countries)

min_age = 0
max_age = 110
gender_data = data_bank["females"]


females_data = data_bank["females"]
males_data = data_bank["males"]



"""
How many percent of data is lost if we truncate maximum age?

For that we look at maximum percent of data lost in countries per year time period.


"""

function country_with_max_data_lost(gender_data::Dict{String, DataFrame}, t::Int, age_truncation::Int)

    deaths_counts_all = group_deaths_count(gender_data, all_countries, t, 0, max_age)
    max_data_lost = 0.0
    arg_max = 1
    for i in 1:size(deaths_counts_all,1)
        data_lost = 1.0 - sum(deaths_counts_all[i, 1:age_truncation+1]) / sum(deaths_counts_all[i,:])
        if data_lost > max_data_lost

            max_data_lost = data_lost
            arg_max = i
        end
    end
    return max_data_lost, arg_max
end



function country_with_max_data_lost(gender_data::Dict{String, DataFrame}, time_periods::Vector{Int}, age_truncation::Int)
    data_lost_per_year = Dict{Int, Tuple}()
    
    for time in time_periods
        data_lost_per_year[time] = country_with_max_data_lost(gender_data, time, age_truncation)
    end
    return data_lost_per_year
end

time_periods = collect(1960:1:2010)
age_truncations = [75, 80,85,95,100]
function save_percentage_data_lost(age_truncations)
    for age in age_truncations
    data_lost = country_with_max_data_lost(data_bank["females"], time_periods, age)

    data_lost = [100.0*data_lost[t][1] for t in time_periods]
    sc = scatter(time_periods, data_lost, xlabel = "Time periods", ylabel = "% lost", ylims = (0.0,100.0),
                            legend = false, title = "Max % lost using truncation at $(age)")
    filepath = joinpath(pwd(), "mortality", "eda_plots")
    filepath = joinpath(filepath, "females_data_lost_$(age).png")
    savefig(sc, filepath)
    end
end

#save_percentage_data_lost(age_truncations)
# println("done: data lost")






"""

Should we do age truncation? 

We should expect pmf to be left-skewed and monotonically decreasing after peak. If this happens in ages 0-100, then we do not truncate.

# Conclusion after plots: do not truncate.

"""

time_periods = [1965,1975, 1992, 2010]

function save_pmf_country(time_periods, country::String)
    min_age = 0
    gender = "males"
    data_gender = data_bank[gender]
    country_index = Bool.(all_countries .== country)
    country_index = collect(1:length(all_countries))[country_index][1]
    
    ages = collect(min_age:1:max_age)

    pl = scatter(title = "PMFs for $(country)", xlabel = "age", ylabel = "pmf")

    for time in time_periods
        _, pmf_all = group_pmf(data_gender, all_countries, time, min_age, max_age)
        pmf_country = pmf_all[country_index,:]
        scatter!(pl, ages, pmf_country, label = "$(time)")
    end
    filepath = joinpath(pwd(), "mortality", "eda_plots")
    filepath = joinpath(filepath, "females_pmf_$(country)_$(min_age)_$(max_age).png")
    savefig(pl, filepath)
end

#pl = save_pmf_country(time_periods, "France")
# println("done: one country pmf plots.")




"""
How do pmfs of counries in two groups differ across time periods?

For that, we have subplot per each time period. In each subplot, we plot pmfs per each group. 
"""

function pmfs_for_timeperiods(gender_data::Dict{String, DataFrame}, group_1::Vector{String}, group_2::Vector{String},
                           time_periods::Vector{Int}, min_age::Int, age_truncation::Int)

    n_years = length(time_periods)
    # list of individual plots
    plot_list = []

    max_val = 0.0

    for (i, t) in enumerate(time_periods)
        @info "Progess: $i / $(n_years)"
        atoms_1, weights_1 = group_pmf(gender_data, group_1, t, min_age, age_truncation)
        _, weights_2 = group_pmf(gender_data, group_2, t, min_age, age_truncation)
        atoms = atoms_1[1,:]

        current_max = maximum((maximum(weights_1), maximum(weights_2)))
        if max_val < current_max
            max_val = current_max
        end
        # Create a subplot for this specific year
        p = scatter(title = "Year $t", titlefontsize=10) 
        for j in 1:size(weights_1, 1)
            lbl = (j == 1) ? "Sov" : ""
            scatter!(p, atoms, weights_1[j,:], color = "green", lw=2, label=lbl, alpha = 0.2)
        end
        for j in 1:size(weights_2, 1)
            lbl = (j == 1) ? "NonSov" : ""
            scatter!(p, atoms, weights_2[j,:], color = "brown", lw=2, label=lbl, alpha = 0.2)
        end 
        push!(plot_list, p)
    end
    final_pl = scatter(plot_list..., layout = (4, 4), size = (1200, 1200),
                    xlabel = "Age", ylabel = "Density", ylims = (0.0,0.06), alpha = 0.2)
   # println("maximum is $(max_val)")
    return final_pl
end


function save_pmfs_for_timeperiods(gender_data::Dict{String, DataFrame}, gender::String, group_1::Vector{String}, group_2::Vector{String},
                           time_periods::Vector{Int}, min_age::Int, age_truncation::Int)

    pl = pmfs_for_timeperiods(gender_data, group_1, group_2, time_periods, min_age, age_truncation)
    filepath = joinpath(pwd(), "mortality", "eda_plots", "pmf_frechet_timeperiods")
    filepath = joinpath(filepath, "$(gender)_allpmfs_$(min_age)_$(max_age).png")
    savefig(pl, filepath)

end

time_periods = collect(1963:3:2010)

save_pmfs_for_timeperiods(females_data, "females", group_1, group_2, time_periods, min_age, max_age)
save_pmfs_for_timeperiods(males_data, "males", group_1, group_2, time_periods, min_age, max_age)

 println("done: pmf plots of all countries per each time period")




"""
Same as previous point, but now we plot quantiles instead of pmfs.
"""

function quantiles_for_timeperiods(gender_data::Dict{String, DataFrame}, group_1::Vector{String}, group_2::Vector{String},
                           time_periods::Vector{Int}, min_age::Int, age_truncation::Int)

    n_years = length(time_periods)
    # list of individual plots
    plot_list = []

    αs = collect(0.0:0.01:1.0)

    for (i, t) in enumerate(time_periods)
        @info "Progess: $i / $(n_years)"
        atoms_1, weights_1 = group_pmf(gender_data, group_1, t, min_age, age_truncation)
        _, weights_2 = group_pmf(gender_data, group_2, t, min_age, age_truncation)
        atoms = atoms_1[1,:]

        
        # Create a subplot for this specific year
        p = plot(title = "Year $t", titlefontsize=10) 
        for j in 1:size(weights_1, 1)
            lbl = (j == 1) ? "Sov" : ""
            pmf_obj = DiscreteNonParametric(atoms, weights_1[j,:])
            q = quantile.(pmf_obj, αs)    
            plot!(p, αs, q, color = "green", lw=2, label=lbl, alpha = 0.5)
        end
        
        for j in 1:size(weights_2, 1)
            lbl = (j == 1) ? "NonSov" : ""
            pmf_obj = DiscreteNonParametric(atoms, weights_2[j,:])
            q = quantile.(pmf_obj, αs)
            plot!(p, αs, q, color = "brown", lw=2, label=lbl, alpha = 0.5)
        end 
        push!(plot_list, p)
    end
    final_pl = plot(plot_list..., layout = (4, 4), size = (1200, 1200),
                    xlabel = "prob level", ylabel = "quantile")
    #println("maximum is $(max_val)")
    return final_pl
end


function save_quantiles_for_timeperiods(gender_data::Dict{String, DataFrame}, gender::String, group_1::Vector{String}, group_2::Vector{String},
                           time_periods::Vector{Int}, min_age::Int, age_truncation::Int)

    pl = quantiles_for_timeperiods(gender_data, group_1, group_2, time_periods, min_age, age_truncation)
    filepath = joinpath(pwd(), "mortality", "eda_plots", "pmf_frechet_timeperiods")
    filepath = joinpath(filepath, "$(gender)_allquantiles_$(min_age)_$(max_age).png")
    savefig(pl, filepath)

end

# save_quantiles_for_timeperiods(females_data, "females", group_1, group_2, time_periods, min_age, max_age)
# save_quantiles_for_timeperiods(males_data, "males", group_1, group_2, time_periods, min_age, max_age)
# println("done: quantiles plots of all countries per each time period")



"""
How do Frechet means differ accross time periods?

As quantile of the Frechet mean in the group is the average of the quentiles in the group, we plot the quantile of FM.
"""
function quantiles_frechet_means(gender_data::Dict{String, DataFrame}, group::Vector{String}, time::Int,
                    min_age::Int, age_truncation::Int, αs::Vector{Float64})
    atoms, weights = group_pmf(gender_data, group, time, min_age, age_truncation)
    n = size(atoms, 1)
    atoms = atoms[1,:]
    quantiles = [quantile.(DiscreteNonParametric(atoms, weights[i,:]), αs) for i in 1:n]
    return vec(mean(quantiles, dims = 1))
end



function quantiles_frechet_means_for_timeperiods(gender_data::Dict{String, DataFrame}, group_1::Vector{String}, group_2::Vector{String},
                                    time_periods::Vector{Int}, min_age::Int, age_truncation::Int)
    plots = []
    αs = collect(0.0:0.01:1.0)
    for time in time_periods
        quantiles_1 = quantiles_frechet_means(gender_data, group_1, time, min_age, age_truncation, αs)
        quantiles_2 = quantiles_frechet_means(gender_data, group_2, time, min_age, age_truncation, αs)
        pl = plot(αs, quantiles_1, xlabel = "prob level", ylabel = "quantile", label = "Sov", color = "green", title = "Quantile of FM")
        plot!(pl, αs , quantiles_2, label = "NonSov", color = "brown")
        push!(plots, pl)
    end
    final_pl = plot(plots..., layout = (4, 4), size = (1200, 1200))
    return final_pl
end




function save_quantiles_frechet_for_time_periods(gender_data::Dict{String, DataFrame}, gender::String,
                group_1::Vector{String}, group_2::Vector{String},
                time_periods::Vector{Int}, min_age::Int, age_truncation::Int)
    pl = quantiles_frechet_means_for_timeperiods(gender_data, group_1, group_2, time_periods, min_age, age_truncation)
    filepath = joinpath(pwd(), "mortality", "eda_plots", "pmf_frechet_timeperiods")
    filepath = joinpath(filepath, "$(gender)_frechet_$(min_age)_$(max_age).png")
    savefig(pl, filepath)
end

time_periods = collect(1963:3:2010)

# save_quantiles_frechet_for_time_periods(females_data, "females", group_1, group_2, time_periods, min_age, max_age)
# save_quantiles_frechet_for_time_periods(males_data, "males", group_1, group_2, time_periods, min_age, max_age)

# println("done: FM quantile plots of all countries per each time period")



"""

If we pool data in groups (obtain mean pmf inside groups), do we observe differences accross groups?

"""
function pooled_pmfs(gender_data::Dict{String, DataFrame}, group::Vector{String}, time::Int,
                    min_age::Int, age_truncation::Int)
    _, weights = group_pmf(gender_data, group, time, min_age, age_truncation)
    return vec(mean(weights, dims = 1))
end



function pooled_pmfs_for_timeperiods(gender_data::Dict{String, DataFrame}, group_1::Vector{String}, group_2::Vector{String},
                                    time_periods::Vector{Int}, min_age::Int, age_truncation::Int)
    plots = []
    ages = collect(min_age:1:age_truncation)
    max_val = 0.0
    for time in time_periods
        pmf_1 = pooled_pmfs(gender_data, group_1, time, min_age, age_truncation)
        pmf_2 = pooled_pmfs(gender_data, group_2, time, min_age, age_truncation)
        current_max = maximum((maximum(pmf_1), maximum(pmf_2)))
        if max_val < current_max
            max_val = current_max
        end
        pl = scatter(ages, pmf_1, xlabel = "age", ylabel = "pmf", label = "Sov", color = "green", title = "Pooled PMFs")
        scatter!(pl, ages , pmf_2, label = "NonSov", color = "brown")
        push!(plots, pl)
    end
    final_pl = scatter(plots..., layout = (4, 4), size = (1200, 1200), ylims = (0.0, 0.06))
    return final_pl
end



function save_pooled_pmfs_for_time_periods(gender_data::Dict{String, DataFrame}, gender::String,
                group_1::Vector{String}, group_2::Vector{String},
                time_periods::Vector{Int}, min_age::Int, age_truncation::Int)
    pl = pooled_pmfs_for_timeperiods(gender_data, group_1, group_2, time_periods, min_age, age_truncation)
    filepath = joinpath(pwd(), "mortality", "eda_plots", "pooled_pmfs")
    filepath = joinpath(filepath, "$(gender)_pooled_pmf_$(min_age)_$(max_age).png")
    savefig(pl, filepath)
end

save_pooled_pmfs_for_time_periods(females_data, "females", group_1, group_2, time_periods, min_age, max_age)
save_pooled_pmfs_for_time_periods(males_data, "males", group_1, group_2, time_periods, min_age, max_age)

println("done: pooled_pmf plots of all countries per each time period")






"""
4 things to consider: pmf, kde, frechet mean, pooled pmf.

1) how do KDEs for each country change over time?   (do maybe subplots on grid)
2) how do frechet means change over time?  (subplots on grid and also intensity change over time)
3) how do pmfs from pooled data change over time? are they same as frechet means? (subplots on grid
                                             and also intensity change over time)
4) obtain p values without pooling on the whole age range
5) obtain p values with pooling on the whole age range. Does it match with 4?
6) obtain p values without pooling on different age bands.

"""



# Exploratory data analysis

# include("data_extractors.jl")
# using RCall
# using Distributions
# using Plots
# using KernelDensity

# function frechet_mean(atoms::Matrix{Float64}, weights::Matrix{Float64})
#     n_countries = size(atoms, 1)
        
#     alphas = collect(0.0:0.01:1.0)
#     quantiles = zeros(n_countries, length(alphas))
    

#     for j in 1:n_countries
#         pmf = DiscreteNonParametric(atoms[j,:], weights[j,:])
#         quantiles[j, :] = quantile.(pmf, alphas)
#     end

#     @rput quantiles n_countries alphas 
#     R"""
#     library(frechet)

#     res = DenFMean(qin = quantiles, optns = list(qSup = alphas))

#     dout = res$dout
#     dsup = res$dSup

#     """
#     @rget dout dsup
#     return dout,dsup
# end

# function frechet_means(atoms_per_year::Array{Float64,3}, weights_per_year::Array{Float64,3})
#     # atoms and weights are 3 dimensional arrays: n_time_periods x n_countries x n_ages

#     n_time_periods = size(atoms_per_year, 1)
#     n_countries = size(atoms_per_year, 2)
        
#     alphas = collect(0.0:0.01:1.0)
#     quantiles_per_year = zeros(n_time_periods, n_countries, length(alphas))
 
#     for i in 1:n_time_periods
#         for j in 1:n_countries
#             pmf = DiscreteNonParametric(atoms_per_year[i, j,:], weights_per_year[i, j,:])
#             quantiles_per_year[i, j, :] = quantile.(Ref(pmf), alphas)
#         end
#     end

#     @rput quantiles_per_year n_time_periods alphas 
#     R"""
#     library(frechet)

#     dout_list <- list()
#     dsup_list <- list()
#     for (i in 1:n_time_periods) {
#         res = DenFMean(qin = quantiles_per_year[i, , ], optns = list(qSup = alphas))
#         dout_list[[i]] = res$dout
#         dsup_list[[i]] = res$dSup
#     }
#     """
#     @rget dout_list dsup_list
#     return dout_list, dsup_list
# end

# function kde_group(atoms::Matrix{Float64}, weights::Matrix{Float64}, min_age::Int, age_truncation::Int)
#     n_countries = size(atoms,1)
#     return [kde(atoms[i,:]; boundary = (min_age, age_truncation), weights = weights[i,:]) for i in 1:n_countries]
# end

# function plot_pmfs(gender_data::Dict{String, DataFrame}, groups::Tuple{Vector{String}, Vector{String}},
#             t::Int, min_age::Int, age_truncation::Int)
#     atoms_1, weights_1 = group_pmf_per_year(gender_data, groups[1], t, min_age, age_truncation)
#     atoms_2, weights_2 = group_pmf_per_year(gender_data, groups[2], t, min_age, age_truncation)
#     color_1, color_2 = "green", "brown"

#     n_countries_1, n_countries_2 = size(atoms_1, 1), size(atoms_2, 1)
    
#     pl = scatter(title = "PMF", xlabel = "age", ylabel = "mass")
#     for i in 1:n_countries_1
#         label = (i == 1) ? "Sov" : ""
#         scatter!(pl, atoms_1[i,:], weights_1[i,:], color = color_1, label = label, alpha = 0.5)
#     end
#     for i in 1:n_countries_2
#         label = (i == 1) ? "West" : ""
#         scatter!(pl, atoms_2[i,:], weights_2[i,:], color = color_2, label = label, alpha = 0.5)
#     end
#     return pl
# end



# function plot_kde(gender_data::Dict{String, DataFrame}, groups::Tuple{Vector{String}, Vector{String}},
#             t::Int, min_age::Int, age_truncation::Int)
#     atoms_1, weights_1 = group_pmf_per_year(gender_data, groups[1], t, min_age, age_truncation)
#     atoms_2, weights_2 = group_pmf_per_year(gender_data, groups[2], t, min_age, age_truncation)
#     color_1, color_2 = "green", "brown"

#     n_countries_1, n_countries_2 = size(atoms_1, 1), size(atoms_2, 1)
    
#     pl = plot(title = "KDE", xlabel = "age", ylabel = "density")
#     for i in 1:n_countries_1
#         label = (i == 1) ? "group1" : ""
#         kde_country = kde(atoms_1[i,:]; boundary = (min_age, age_truncation), weights = weights_1[i,:])
#         plot!(pl, kde_country.x, kde_country.density, color = color_1, label = label, alpha = 0.5)
#     end
#     for i in 1:n_countries_2
#         label = (i == 1) ? "group2" : ""
#         kde_country = kde(atoms_2[i,:]; boundary = (min_age, age_truncation), weights = weights_2[i,:])
#         plot!(pl, kde_country.x, kde_country.density, color = color_2, label = label, alpha = 0.5)
#     end
#     return pl
# end

# function plot_frechetmean(gender_data::Dict{String, DataFrame}, groups::Tuple{Vector{String}, Vector{String}},
#             t::Int, min_age::Int, age_truncation::Int)
#     atoms_1, weights_1 = group_pmf_per_year(gender_data, groups[1], t, min_age, age_truncation)
#     atoms_2, weights_2 = group_pmf_per_year(gender_data, groups[2], t, min_age, age_truncation)


#     dout_1, dsup_1 = frechet_mean(atoms_1, weights_1)
#     dout_2, dsup_2 = frechet_mean(atoms_2, weights_2)
    
#     pl = plot(title = "Frechet mean", xlabel = "age", ylabel = "density")
#     plot!(pl, dsup_1, dout_1, label = "group1", color = "green")
#     plot!(pl, dsup_2, dout_2, label = "group2", color = "brown")
#     return pl
# end

# function plot_frechetmeans(gender_data::Dict{String, DataFrame}, groups::Tuple{Vector{String}, Vector{String}},
#             time_periods::Vector{Int}, min_age::Int, age_truncation::Int)

#     pl = plot(title = "Frechet mean", xlabel = "age", ylabel = "density", ylims = (0.0, 0.07))
#     n_years = length(time_periods)

#     # collect all atoms and weights
#     atoms_per_year_1 = zeros(n_years, length(groups[1]), age_truncation - min_age + 1)
#     weights_per_year_1 = zeros(n_years, length(groups[1]), age_truncation - min_age + 1)

#     atoms_per_year_2 = zeros(n_years, length(groups[2]), age_truncation - min_age + 1)
#     weights_per_year_2 = zeros(n_years, length(groups[2]), age_truncation - min_age + 1)

#     for (i, t) in enumerate(time_periods)
#         atoms_per_year_1[i, :, :], weights_per_year_1[i, :, :] = group_pmf_per_year(gender_data, groups[1], t, min_age, age_truncation)
#         atoms_per_year_2[i, :, :], weights_per_year_2[i, :, :] = group_pmf_per_year(gender_data, groups[2], t, min_age, age_truncation)
        
#     end
#     dout_1, dsup_1 = frechet_means(atoms_per_year_1, weights_per_year_1)
#     dout_2, dsup_2 = frechet_means(atoms_per_year_2, weights_per_year_2)

#     for i in 1:n_years
#         intensity = 0.07 + (0.95 * (i / n_years))
#         label = i == 1 ? "group1" : ""
#         plot!(pl, dsup_1[i], dout_1[i], color = "green", label = label, alpha = intensity)
#         label = i == 1 ? "group2" : ""
#         plot!(pl, dsup_2[i], dout_2[i], color = "brown", label = label, alpha = intensity)
#     end
#     return pl
# end

# function plot_frechet_grid(gender_data::Dict{String, DataFrame}, 
#                            groups::Tuple{Vector{String}, Vector{String}},
#                            time_periods::Vector{Int}, min_age::Int, age_truncation::Int)
#     n_years = length(time_periods)

#     # collect all atoms and weights
#     atoms_per_year_1 = zeros(n_years, length(groups[1]), age_truncation - min_age + 1)
#     weights_per_year_1 = zeros(n_years, length(groups[1]), age_truncation - min_age + 1)

#     atoms_per_year_2 = zeros(n_years, length(groups[2]), age_truncation - min_age + 1)
#     weights_per_year_2 = zeros(n_years, length(groups[2]), age_truncation - min_age + 1)

#     for (i, t) in enumerate(time_periods)
#         atoms_per_year_1[i, :, :], weights_per_year_1[i, :, :] = group_pmf_per_year(gender_data, groups[1], t, min_age, age_truncation)
#         atoms_per_year_2[i, :, :], weights_per_year_2[i, :, :] = group_pmf_per_year(gender_data, groups[2], t, min_age, age_truncation)
#     end
#     dout_1, dsup_1 = frechet_means(atoms_per_year_1, weights_per_year_1)
#     dout_2, dsup_2 = frechet_means(atoms_per_year_2, weights_per_year_2)

#     plot_list = []
#     for (i, t) in enumerate(time_periods)

#         # Create a subplot for this specific year
#         p = plot(title = "Year $t", titlefontsize=10, ylims = (0.0, 0.07)) 

#         plot!(p, dsup_1[i], dout_1[i], color = "green", lw=2, label="group1")
#         plot!(p, dsup_2[i], dout_2[i], color = "brown", lw=2, label="group2")
        
#         push!(plot_list, p)
#     end
#     final_pl = plot(plot_list..., layout = (4, 4), size = (1200, 1200),
#                     xlabel = "Age", ylabel = "Density", ylims = (0.0, 0.07))
#     return final_pl
# end

# function plot_pooled_pmfs(gender_data::Dict{String, DataFrame}, groups::Tuple{Vector{String}, Vector{String}},
#             t::Int, min_age::Int, age_truncation::Int)
#     atoms_1, weights_1 = group_pmf_per_year(gender_data, groups[1], t, min_age, age_truncation)
#     atoms_2, weights_2 = group_pmf_per_year(gender_data, groups[2], t, min_age, age_truncation)

#     atoms_1 = atoms_1[1,:]
#     atoms_2 = atoms_1[:]
#     weights_1 = vec(mean(weights_1, dims = 1))
#     weights_2 = vec(mean(weights_2, dims = 1))

#     pl = scatter(title = "PMF", xlabel = "age", ylabel = "mass")
    
#     scatter!(pl, atoms_1, weights_1, color = "green", label = "group1", alpha = 0.5)
#     scatter!(pl, atoms_2, weights_2, color = "brown", label = "group2", alpha = 0.5)
#     return pl
# end



# function plot_kdes_grid(gender_data::Dict{String, DataFrame}, 
#                            groups::Tuple{Vector{String}, Vector{String}},
#                            time_periods::Vector{Int}, min_age::Int, age_truncation::Int)

    
#     n_years = length(time_periods)
    
#     # list of individual plots
#     plot_list = []

#     for (i, t) in enumerate(time_periods)
#         @info "Progess: $i / $(n_years)"
#         atoms_1, weights_1 = group_pmf_per_year(gender_data, groups[1], t, min_age, age_truncation)
#         atoms_2, weights_2 = group_pmf_per_year(gender_data, groups[2], t, min_age, age_truncation)

#         kde_1 = kde_group(atoms_1, weights_1, min_age, age_truncation)
#         kde_2 = kde_group(atoms_2, weights_2, min_age, age_truncation)
        
        
#         # Create a subplot for this specific year
#         p = plot(title = "Year $t", titlefontsize=10, ylims = (0.0, 0.07)) 
#         for j in 1:length(kde_1)
#             lbl = (j == 1) ? "group1" : ""
#             plot!(p, kde_1[j].x, kde_1[j].density, color = "green", lw=2, label=lbl, alpha = 0.5)
#         end
#         for j in 1:length(kde_2)
#             lbl = (j == 1) ? "group2" : ""
#             plot!(p, kde_2[j].x, kde_2[j].density, color = "brown", lw=2, label=lbl, alpha = 0.5)
#         end 
#         push!(plot_list, p)
#     end

    
#     final_pl = plot(plot_list..., layout = (4, 4), size = (1200, 1200),
#                     xlabel = "Age", ylabel = "Density", ylims = (0.0, 0.07),)

#     return final_pl
# end


# group1 = ["belarus", "Bulgaria", "Czechia", "Estonia", "Hungary", "Latvia", "Poland", "Lithuania", "Russia", "Slovakia", "Ukraine"]

# group2 = ["Australia", "Austria", "Belgium", "Canada", "Denmark", "Finland", "France", "Iceland", "Ireland", "Italy", 
# "Japan", "Luxembourg", "Netherlands", "NewZealand", "Norway", "Spain", "Sweden",
# "Switzerland", "UnitedKingdom" , "UnitedStatesofAmerica"]



# group_config = [(group1, 1), (group2, 2)]
# genders = ["males", "females"]
# data_bank = load_mortality_data(group_config, genders)


# min_age = 0
# age_truncation = 85
# t = 1960

# gender = "males"
# # pl_pmfs = plot_pmfs(data_bank[gender], (group1, group2), t, min_age, age_truncation)
# # pl_kdes = plot_kde(data_bank[gender], (group1, group2), t, min_age, age_truncation)

# # pl_frechet = plot_frechetmean(data_bank[gender], (group1, group2), t, min_age, age_truncation)
# # pl_pooled = plot_pooled_pmfs(data_bank[gender], (group1, group2), t, min_age, age_truncation)

# time_periods = collect(1963:3:2010)

# function save_plots(time_periods, gender)
#     pl_frechets_grid = plot_frechet_grid(data_bank[gender], (group1, group2), time_periods, min_age, age_truncation)
#     pl_frechetmeans = plot_frechetmeans(data_bank[gender], (group1, group2), time_periods, min_age, age_truncation)
#     pl_kdes = plot_kdes_grid(data_bank[gender], (group1, group2), time_periods, min_age, age_truncation)
#     output_dir = joinpath(pwd(), "applications", "plots")
#     mkpath(output_dir)
#     savefig(pl_kdes, joinpath(output_dir, "$(gender)_kdes.png"))
#     savefig(pl_frechets_grid, joinpath(output_dir, "$(gender)_frechets_grid.png"))
#     savefig(pl_frechetmeans, joinpath(output_dir, "$(gender)_frechets.png"))
# end
# save_plots(time_periods, "males")
# save_plots(time_periods, "females")
# print("done")







# # x = rand(10,3000)

# # kd = [kde(x[i,:]) for i in 1:10]

# # pl = plot(ylims = (0.0,1.5))
# # for i in 1:10
# #     plot!(pl, kd[i].x, kd[i].density)
# # end
# # pl