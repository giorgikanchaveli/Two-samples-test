include("data_extractors.jl")



group_1 = ["belarus", "Bulgaria", "Czechia", "Estonia", "Hungary", "Latvia", "Poland", "Lithuania", "Russia", "Slovakia", "Ukraine"]
group_2 = ["Australia", "Austria", "Belgium", "Canada", "Denmark", "Finland", "France", "Iceland", "Ireland", "Italy", 
"Japan", "Luxembourg", "Netherlands", "NewZealand", "Norway", "Spain", "Sweden",
"Switzerland", "UnitedKingdom" , "UnitedStatesofAmerica"]
all_countries = vcat(group_1, group_2)

data_bank = load_mortality_data(all_countries)
females_data = data_bank["females"]
males_data = data_bank["males"]
gender_data = data_bank["males"]


time = 2010
min_age = 0
age_truncation = 110

observations_1 = pool_group_deaths_count(group_deaths_count(gender_data, group_1, time, min_age, age_truncation))
observations_2 = pool_group_deaths_count(group_deaths_count(gender_data, group_2, time, min_age, age_truncation))
pooled_distance = wasserstein_1d_general(sort(observations_1), sort(observations_2))


atoms_1, weights_1 = group_pmf(gender_data, group_1, time, min_age, age_truncation)
atoms_2, weights_2 = group_pmf(gender_data, group_2, time, min_age, age_truncation)
average_weights_1 = vec(mean(weights_1, dims = 1))
average_weights_2 = vec(mean(weights_2, dims = 1))
averaged_distance = sum(abs.(cumsum(average_weights_1) .- cumsum(average_weights_2)))

println(averaged_distance)
println(pooled_distance)
println("diff: $(abs(averaged_distance-pooled_distance))")


