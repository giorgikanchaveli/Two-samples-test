# In this file we do Two sample test for two groups of countries per each time period.

using Plots

include("../methods.jl")
include("data_extractors.jl")

function pvalues_hipm(gender_data::Dict{String, DataFrame}, group_1::Vector{String}, group_2::Vector{String},
                 time_periods::Vector{Int}, min_age::Int, age_truncation::Int,
                 n_permutations::Int, max_time::Float64, pooled::Bool) 
    n_years = length(time_periods) 
    pvalues = Vector{Float64}(undef, n_years) # store p values per each time

    for (i, time) in enumerate(time_periods)
        @info "Progess $i / $(n_years)"
        atoms_1, weights_1 = group_pmf(gender_data, group_1, time, min_age, age_truncation)
        atoms_2, weights_2 = group_pmf(gender_data, group_2, time, min_age, age_truncation)
        pvalues[i] = pvalue_hipm(atoms_1, weights_1, atoms_2, weights_2, n_permutations, max_time, pooled)
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

data_bank = load_mortality_data(["males", "females"], all_countries)
females_data = data_bank["females"]
males_data = data_bank["males"]


min_age = 0
max_age = 110
time_periods = collect(1960:1:2010)


n_permutations = 100
max_time = 1.0

pvalues_males_notpooled = pvalues_hipm(males_data, group_1, group_2, time_periods, min_age, max_age, n_permutations, max_time, false)
pvalues_males_pooled = pvalues_hipm(males_data, group_1, group_2, time_periods, min_age, max_age, n_permutations, max_time, true)

save_pvalues(pvalues_males_notpooled, time_periods, 
             "P-values, males, Not Pooled", "pvalues_males_not_pooled")
save_pvalues(pvalues_males_pooled, time_periods, 
             "P-values, males, Pooled", "pvalues_males_pooled")

pvalues_females_notpooled = pvalues_hipm(females_data, group_1, group_2, time_periods, min_age, max_age, n_permutations, max_time, false)
pvalues_females_pooled = pvalues_hipm(females_data, group_1, group_2, time_periods, min_age, max_age, n_permutations, max_time, true)

save_pvalues(pvalues_females_notpooled, time_periods, 
             "P-values, females, Not Pooled", "pvalues_females_not_pooled")
save_pvalues(pvalues_females_pooled, time_periods, 
             "P-values, females, Pooled", "pvalues_females_pooled")


