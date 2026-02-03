# To do


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

include("mortality.jl")
using RCall
using Distributions
using Plots
using KernelDensity

function frechet_mean(atoms::Matrix{Float64}, weights::Matrix{Float64})
    n_countries = size(atoms, 1)
        
    alphas = collect(0.0:0.01:1.0)
    quantiles = zeros(n_countries, length(alphas))
    

    for j in 1:n_countries
        pmf = DiscreteNonParametric(atoms[j,:], weights[j,:])
        quantiles[j, :] = quantile.(pmf, alphas)
    end

    @rput quantiles n_countries alphas 
    R"""
    library(frechet)

    res = DenFMean(qin = quantiles, optns = list(qSup = alphas))

    dout = res$dout
    dsup = res$dSup

    """
    @rget dout dsup
    return dout,dsup
end

function plot_pmfs(gender_data::Dict{String, DataFrame}, groups::Tuple{Vector{String}, Vector{String}},
            t::Int, min_age::Int, max_age::Int)
    atoms_1, weights_1 = group_pmf_per_year(gender_data, groups[1], t, min_age, max_age)
    atoms_2, weights_2 = group_pmf_per_year(gender_data, groups[2], t, min_age, max_age)
    color_1, color_2 = "green", "brown"

    n_countries_1, n_countries_2 = size(atoms_1, 1), size(atoms_2, 1)
    
    pl = scatter(title = "PMF", xlabel = "age", ylabel = "mass")
    for i in 1:n_countries_1
        label = (i == 1) ? "group1" : ""
        scatter!(pl, atoms_1[i,:], weights_1[i,:], color = color_1, label = label, alpha = 0.5)
    end
    for i in 1:n_countries_2
        label = (i == 1) ? "group2" : ""
        scatter!(pl, atoms_2[i,:], weights_2[i,:], color = color_2, label = label, alpha = 0.5)
    end
    return pl
end

function plot_kde(gender_data::Dict{String, DataFrame}, groups::Tuple{Vector{String}, Vector{String}},
            t::Int, min_age::Int, max_age::Int)
    atoms_1, weights_1 = group_pmf_per_year(gender_data, groups[1], t, min_age, max_age)
    atoms_2, weights_2 = group_pmf_per_year(gender_data, groups[2], t, min_age, max_age)
    color_1, color_2 = "green", "brown"

    n_countries_1, n_countries_2 = size(atoms_1, 1), size(atoms_2, 1)
    
    pl = scatter(title = "KDE", xlabel = "age", ylabel = "density")
    for i in 1:n_countries_1
        label = (i == 1) ? "group1" : ""
        kde_country = kde(atoms_1[i,:]; boundary = (min_age, max_age), weights = weights_1[i,:])
        plot!(pl, kde_country.x, kde_country.density, color = color_1, label = label, alpha = 0.5)
    end
    for i in 1:n_countries_2
        label = (i == 1) ? "group2" : ""
        kde_country = kde(atoms_2[i,:]; boundary = (min_age, max_age), weights = weights_2[i,:])
        plot!(pl, kde_country.x, kde_country.density, color = color_2, label = label, alpha = 0.5)
    end
    return pl
end

function plot_frechetmean(gender_data::Dict{String, DataFrame}, groups::Tuple{Vector{String}, Vector{String}},
            t::Int, min_age::Int, max_age::Int)
    atoms_1, weights_1 = group_pmf_per_year(gender_data, groups[1], t, min_age, max_age)
    atoms_2, weights_2 = group_pmf_per_year(gender_data, groups[2], t, min_age, max_age)


    dout_1, dsup_1 = frechet_mean(atoms_1, weights_1)
    dout_2, dsup_2 = frechet_mean(atoms_2, weights_2)
    
    pl = plot(title = "Frechet mean", xlabel = "age", ylabel = "density")
    plot!(pl, dsup_1, dout_1, label = "group1", color = "green")
    plot!(pl, dsup_2, dout_2, label = "group2", color = "brown")
    return pl
end

function plot_frechetmean(gender_data::Dict{String, DataFrame}, groups::Tuple{Vector{String}, Vector{String}},
            time_periods::Vector{Int}, min_age::Int, max_age::Int)

    pl = plot(title = "Frechet mean", xlabel = "age", ylabel = "density")
    n_years = length(time_periods)

    for (i, t) in enumerate(time_periods)
        @info "Progress: $i / $(n_years)"
        atoms_1, weights_1 = group_pmf_per_year(gender_data, groups[1], t, min_age, max_age)
        atoms_2, weights_2 = group_pmf_per_year(gender_data, groups[2], t, min_age, max_age)
        color_1, color_2 = "green", "brown"

        dout_1, dsup_1 = frechet_mean(atoms_1, weights_1)
        dout_2, dsup_2 = frechet_mean(atoms_2, weights_2)
        intensity = 0.2 + (0.8 * (i / n_years))
        label = i == 1 ? "group1" : ""
        plot!(pl, dsup_1, dout_1, color = color_1, label = label, alpha = intensity)
        label = i == 1 ? "group2" : ""
        plot!(pl, dsup_2, dout_2, color = color_2, label = label, alpha = intensity)
    end
    return pl
end
function plot_frechet_grid(gender_data::Dict{String, DataFrame}, 
                           groups::Tuple{Vector{String}, Vector{String}},
                           time_periods::Vector{Int}, min_age::Int, max_age::Int)

    
    n_years = length(time_periods)
    
    # list of individual plots
    plot_list = []

    for (i, t) in enumerate(time_periods)
        @info "Progess: $i / $(n_years)"
        atoms_1, weights_1 = group_pmf_per_year(gender_data, groups[1], t, min_age, max_age)
        atoms_2, weights_2 = group_pmf_per_year(gender_data, groups[2], t, min_age, max_age)

        dout_1, dsup_1 = frechet_mean(atoms_1, weights_1)
        dout_2, dsup_2 = frechet_mean(atoms_2, weights_2)

        
        # Create a subplot for this specific year
        p = plot(title = "Year $t", titlefontsize=10, ylims = (0, 0.1)) 

        plot!(p, dsup_1, dout_1, color = "green", lw=2, label="group1")
        plot!(p, dsup_2, dout_2, color = "brown", lw=2, label="group2")
        
        push!(plot_list, p)
    end

    
    final_pl = plot(plot_list..., layout = (3, 3), size = (1200, 1200),
                    xlabel = "Age", ylabel = "Density")

    return final_pl
end

function plot_pooled_pmfs(gender_data::Dict{String, DataFrame}, groups::Tuple{Vector{String}, Vector{String}},
            t::Int, min_age::Int, max_age::Int)
    atoms_1, weights_1 = group_pmf_per_year(gender_data, groups[1], t, min_age, max_age)
    atoms_2, weights_2 = group_pmf_per_year(gender_data, groups[2], t, min_age, max_age)

    atoms_1 = atoms_1[1,:]
    atoms_2 = atoms_1[:]
    weights_1 = vec(mean(weights_1, dims = 1))
    weights_2 = vec(mean(weights_2, dims = 1))

    pl = scatter(title = "PMF", xlabel = "age", ylabel = "mass")
    
    scatter!(pl, atoms_1, weights_1, color = "green", label = "group1", alpha = 0.5)
    scatter!(pl, atoms_2, weights_2, color = "brown", label = "group2", alpha = 0.5)
    return pl
end

group1 = ["belarus", "Bulgaria", "Czechia", "Estonia", "Hungary", "Latvia", "Poland", "Lithuania", "Russia", "Slovakia", "Ukraine"]

group2 = ["Australia", "Austria", "Belgium", "Canada", "Denmark", "Finland", "France", "Iceland", "Ireland", "Italy", 
"Japan", "Luxembourg", "Netherlands", "NewZealand", "Norway", "Spain", "Sweden",
"Switzerland", "UnitedKingdom" , "UnitedStatesofAmerica"]



group_config = [(group1, 1), (group2, 2)]
genders = ["males", "females"]
data_bank = load_mortality_data(group_config, genders)


min_age = 0
max_age = 85
t = 1960

gender = "males"
# pl_pmfs = plot_pmfs(data_bank[gender], (group1, group2), t, min_age, max_age)
# pl_kdes = plot_kde(data_bank[gender], (group1, group2), t, min_age, max_age)

# pl_frechet = plot_frechetmean(data_bank[gender], (group1, group2), t, min_age, max_age)
pl_pooled = plot_pooled_pmfs(data_bank[gender], (group1, group2), t, min_age, max_age)

time_periods = collect(1960:6:2010)
# pl_frechets = plot_frechetmean(data_bank[gender], (group1, group2), time_periods, min_age, max_age)
# pl_new = plot_frechet_grid(data_bank[gender],  (group1, group2), time_periods, min_age, max_age)
print("done")

