using CSV, DataFrames


"""
    load_mortality_data

Loads all data into memory. We store DataFrame per each gender and country.

# Arguments

    country_names::Vector{String}
"""
function load_mortality_data(country_names::Vector{String})
    # Structure: data_bank[gender][country_name] = DataFrame
    data_bank = Dict{String, Dict{String, DataFrame}}()
    genders = ["males", "females"]
    for gender in genders
        data_bank[gender] = Dict{String, DataFrame}()

        filepath = joinpath(pwd(), "mortality", "dataset", gender)
        for country in country_names
            fullpath = joinpath(filepath, "$(country)_$(gender).txt")
            if isfile(fullpath)
                df = open(fullpath) do io
                    readline(io) # skip metadata
                    CSV.read(io, DataFrame; delim=' ', ignorerepeated=true)
                end
                data_bank[gender][country] = df
            else
                @warn "File not found: $fullpath"
            end
        end
    end
    return data_bank
end


"""
    country_deaths_count

Given dataframe for some country, returns number of deaths per ages from min_age to max_age.

# Arguments:
    df::DataFrame
    t::Int          :  Year
    min_age::Int
    max_age::Int
"""
function country_deaths_count(df::DataFrame, t::Int, min_age::Int, max_age::Int)
    max_age < 111 || throw(ArgumentError("maximum age must be lower than 111."))
    row_idx = findfirst(==(t), df[!, :Year])
    @assert row_idx !== nothing "Year $t not found."

    dx_band = df[row_idx + min_age:row_idx + max_age, :dx]

    if eltype(dx_band) <: AbstractString
        dx_band = parse.(Float64, dx_band)
    end
    return dx_band
end


"""
    group_deaths_count

Given specific year, collects number of deaths at ages per each country in the group.

# Arguments:
    gender_data::Dict{String, DataFrame}
    group::Vector{String}
    t::Int  :  Year
    min_age::Int
    max_age::Int
"""
function group_deaths_count(gender_data::Dict{String, DataFrame}, group::Vector{String},
                                    t::Int, min_age::Int, max_age::Int)
    min_age < max_age || throw(ArgumentError("minimum age must be strictly smaller than maximum age."))
    deaths_count = Matrix{Float64}(undef, length(group), max_age - min_age + 1)

    for i in 1:length(group)
        country_df = gender_data[group[i]]
        deaths_count[i, :] .= country_deaths_count(country_df, t, min_age, max_age)
    end
    return deaths_count
end


"""
    hier_sample_from_counts

Given matrix of deaths counts per each country, create observations of ages at deaths for each country. This way we obtain
hierarchical sample where rows might have different length.

# Arguments:
    deaths_count::Matrix{Float64}
"""
function hier_sample_from_counts(deaths_count::Matrix{Float64})
    n_rows, n_columns = size(deaths_count)
    row_lengths = Int.(vec(sum(deaths_count, dims = 2)))
    hier_sample = Vector{Vector{Float64}}()
    for i in 1:n_rows
        observations = Vector{Float64}(undef, row_lengths[i])
        index = 1
        for j in 1:n_columns
            n_obs_at_age = Int(deaths_count[i, j])
            observations[index:index+n_obs_at_age-1] .= Float64(j-1)
            index = index + n_obs_at_age
        end
        push!(hier_sample, observations)
    end
    return hier_sample
end


"""
    pool_group_deaths_count

Given matrix of deaths counts per each country, create observations of ages at deaths pooled from each country.

# Arguments:
    deaths_count::Matrix{Float64}
"""
function pool_group_deaths_count(deaths_count::Matrix{Float64})
    n_total = Int(sum(deaths_count))
    pooled_observations = Vector{Float64}(undef, n_total)
    n, m = size(deaths_count) # n refers to countries, m refers to ages
    index = 1
    for i in 1:n
        for j in 1:m
            count = Int(deaths_count[i, j])
            pooled_observations[index:index + count - 1] .= Float64(j-1)
            index = index + count
        end
    end
    return pooled_observations
end


"""
    group_pmf

Given specific year, collects pmfs of deaths at ages per each country in the group.

# Arguments:
    gender_data::Dict{String, DataFrame}
    group::Vector{String}
    t::Int  :  Year
    min_age::Int
    max_age::Int
"""
function group_pmf(gender_data::Dict{String, DataFrame}, group::Vector{String},
                                    t::Int, min_age::Int, max_age::Int)
    min_age < max_age || throw(ArgumentError("minimum age must be strictly smaller than maximum age."))

    ages = Float64.(collect(min_age:max_age))
    atoms = repeat(ages', length(group), 1)

    deaths_count = group_deaths_count(gender_data, group, t, min_age, max_age)
    weights = deaths_count ./ sum(deaths_count, dims = 2)

    return atoms, weights
end


"""
    group_infant_pmf

Obtains pmf for infant death rate per each country. Per each country we have Bernoulli distribution. This
is represented by atoms [0, 1] and weights associated to it.

# Arguments:
    gender_data::Dict{String, DataFrame}
    group::Vector{String}
    t::Int  :  Year
    max_age::Int
"""
function group_infant_pmf(gender_data::Dict{String, DataFrame}, group::Vector{String}, t::Int, max_age::Int)

    # 0 = survival, 1 = death
    atoms = repeat([0.0, 1.0]', length(group), 1)
    weights = Matrix{Float64}(undef, length(group), 2)

    for i in 1:length(group)
        df = gender_data[group[i]]
        row_idx = findfirst(==(t), df[!, :Year])
        @assert row_idx !== nothing "Year $t not found for $(group[i])"

        dx_infant = df[row_idx, :dx]
        if dx_infant isa AbstractString
            dx_infant = parse(Float64, dx_infant)
        else
            dx_infant = Float64(dx_infant)
        end

        dx_band = df[row_idx:(row_idx + max_age), :dx]
        if eltype(dx_band) <: AbstractString
            dx_band = parse.(Float64, dx_band)
        end
        total_deaths = sum(dx_band)

        weights[i, 2] = dx_infant / total_deaths
        weights[i, 1] = 1.0 - weights[i, 2]
    end
    return atoms, weights
end
