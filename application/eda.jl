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

function frechet_means(atoms_per_year::Array{Float64,3}, weights_per_year::Array{Float64,3})
    # atoms and weights are 3 dimensional arrays: n_time_periods x n_countries x n_ages

    n_time_periods = size(atoms_per_year, 1)
    n_countries = size(atoms_per_year, 2)
        
    alphas = collect(0.0:0.01:1.0)
    quantiles_per_year = zeros(n_time_periods, n_countries, length(alphas))
 
    for i in 1:n_time_periods
        for j in 1:n_countries
            pmf = DiscreteNonParametric(atoms_per_year[i, j,:], weights_per_year[i, j,:])
            quantiles_per_year[i, j, :] = quantile.(Ref(pmf), alphas)
        end
    end

    @rput quantiles_per_year n_time_periods alphas 
    R"""
    library(frechet)

    dout_list <- list()
    dsup_list <- list()
    for (i in 1:n_time_periods) {
        res = DenFMean(qin = quantiles_per_year[i, , ], optns = list(qSup = alphas))
        dout_list[[i]] = res$dout
        dsup_list[[i]] = res$dSup
    }
    """
    @rget dout_list dsup_list
    return dout_list, dsup_list
end

function kde_group(atoms::Matrix{Float64}, weights::Matrix{Float64}, min_age::Int, max_age::Int)
    n_countries = size(atoms,1)
    return [kde(atoms[i,:]; boundary = (min_age, max_age), weights = weights[i,:]) for i in 1:n_countries]
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
    
    pl = plot(title = "KDE", xlabel = "age", ylabel = "density")
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

function plot_frechetmeans(gender_data::Dict{String, DataFrame}, groups::Tuple{Vector{String}, Vector{String}},
            time_periods::Vector{Int}, min_age::Int, max_age::Int)

    pl = plot(title = "Frechet mean", xlabel = "age", ylabel = "density", ylims = (0.0, 0.07))
    n_years = length(time_periods)

    # collect all atoms and weights
    atoms_per_year_1 = zeros(n_years, length(groups[1]), max_age - min_age + 1)
    weights_per_year_1 = zeros(n_years, length(groups[1]), max_age - min_age + 1)

    atoms_per_year_2 = zeros(n_years, length(groups[2]), max_age - min_age + 1)
    weights_per_year_2 = zeros(n_years, length(groups[2]), max_age - min_age + 1)

    for (i, t) in enumerate(time_periods)
        atoms_per_year_1[i, :, :], weights_per_year_1[i, :, :] = group_pmf_per_year(gender_data, groups[1], t, min_age, max_age)
        atoms_per_year_2[i, :, :], weights_per_year_2[i, :, :] = group_pmf_per_year(gender_data, groups[2], t, min_age, max_age)
        
    end
    dout_1, dsup_1 = frechet_means(atoms_per_year_1, weights_per_year_1)
    dout_2, dsup_2 = frechet_means(atoms_per_year_2, weights_per_year_2)

    for i in 1:n_years
        intensity = 0.07 + (0.95 * (i / n_years))
        label = i == 1 ? "group1" : ""
        plot!(pl, dsup_1[i], dout_1[i], color = "green", label = label, alpha = intensity)
        label = i == 1 ? "group2" : ""
        plot!(pl, dsup_2[i], dout_2[i], color = "brown", label = label, alpha = intensity)
    end
    return pl
end

function plot_frechet_grid(gender_data::Dict{String, DataFrame}, 
                           groups::Tuple{Vector{String}, Vector{String}},
                           time_periods::Vector{Int}, min_age::Int, max_age::Int)
    n_years = length(time_periods)

    # collect all atoms and weights
    atoms_per_year_1 = zeros(n_years, length(groups[1]), max_age - min_age + 1)
    weights_per_year_1 = zeros(n_years, length(groups[1]), max_age - min_age + 1)

    atoms_per_year_2 = zeros(n_years, length(groups[2]), max_age - min_age + 1)
    weights_per_year_2 = zeros(n_years, length(groups[2]), max_age - min_age + 1)

    for (i, t) in enumerate(time_periods)
        atoms_per_year_1[i, :, :], weights_per_year_1[i, :, :] = group_pmf_per_year(gender_data, groups[1], t, min_age, max_age)
        atoms_per_year_2[i, :, :], weights_per_year_2[i, :, :] = group_pmf_per_year(gender_data, groups[2], t, min_age, max_age)
    end
    dout_1, dsup_1 = frechet_means(atoms_per_year_1, weights_per_year_1)
    dout_2, dsup_2 = frechet_means(atoms_per_year_2, weights_per_year_2)

    plot_list = []
    for (i, t) in enumerate(time_periods)

        # Create a subplot for this specific year
        p = plot(title = "Year $t", titlefontsize=10, ylims = (0.0, 0.07)) 

        plot!(p, dsup_1[i], dout_1[i], color = "green", lw=2, label="group1")
        plot!(p, dsup_2[i], dout_2[i], color = "brown", lw=2, label="group2")
        
        push!(plot_list, p)
    end
    final_pl = plot(plot_list..., layout = (4, 4), size = (1200, 1200),
                    xlabel = "Age", ylabel = "Density", ylims = (0.0, 0.07))
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



function plot_pooled_kdes_grid(gender_data::Dict{String, DataFrame}, 
                           groups::Tuple{Vector{String}, Vector{String}},
                           time_periods::Vector{Int}, min_age::Int, max_age::Int)

    
    n_years = length(time_periods)
    
    # list of individual plots
    plot_list = []

    for (i, t) in enumerate(time_periods)
        @info "Progess: $i / $(n_years)"
        atoms_1, weights_1 = group_pmf_per_year(gender_data, groups[1], t, min_age, max_age)
        atoms_2, weights_2 = group_pmf_per_year(gender_data, groups[2], t, min_age, max_age)

        kde_1 = kde_group(atoms_1, weights_1, min_age, max_age)
        kde_2 = kde_group(atoms_2, weights_2, min_age, max_age)
        
        
        # Create a subplot for this specific year
        p = plot(title = "Year $t", titlefontsize=10, ylims = (0.0, 0.07)) 
        for j in 1:length(kde_1)
            lbl = (j == 1) ? "group1" : ""
            plot!(p, kde_1[j].x, kde_1[j].density, color = "green", lw=2, label=lbl, alpha = 0.5)
        end
        for j in 1:length(kde_2)
            lbl = (j == 1) ? "group2" : ""
            plot!(p, kde_2[j].x, kde_2[j].density, color = "brown", lw=2, label=lbl, alpha = 0.5)
        end 
        push!(plot_list, p)
    end

    
    final_pl = plot(plot_list..., layout = (4, 4), size = (1200, 1200),
                    xlabel = "Age", ylabel = "Density", ylims = (0.0, 0.07),)

    return final_pl
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
# pl_pooled = plot_pooled_pmfs(data_bank[gender], (group1, group2), t, min_age, max_age)

time_periods = collect(1963:3:2010)

function save_plots(time_periods, gender)
    pl_frechets_grid = plot_frechet_grid(data_bank[gender], (group1, group2), time_periods, min_age, max_age)
    pl_frechetmeans = plot_frechetmeans(data_bank[gender], (group1, group2), time_periods, min_age, max_age)
    pl_pooled_kdes = plot_pooled_kdes_grid(data_bank[gender], (group1, group2), time_periods, min_age, max_age)
    output_dir = joinpath(pwd(), "applications", "plots")
    mkpath(output_dir)
    savefig(pl_pooled_kdes, joinpath(output_dir, "$(gender)_pooled_kdes.png"))
    savefig(pl_frechets_grid, joinpath(output_dir, "$(gender)_frechets_grid.png"))
    savefig(pl_frechetmeans, joinpath(output_dir, "$(gender)_frechets.png"))
end
save_plots(time_periods, "males")
save_plots(time_periods, "females")
print("done")







# x = rand(10,3000)

# kd = [kde(x[i,:]) for i in 1:10]

# pl = plot(ylims = (0.0,1.5))
# for i in 1:10
#     plot!(pl, kd[i].x, kd[i].density)
# end
# pl