using CSV, DataFrames

group1 = ["belarus", "Bulgaria", "Czechia", "Estonia", "Hungary", "Latvia", "Poland", "Lithuania", "Russia", "Slovakia", "Ukraine"]

group2 = ["Australia", "Austria", "Belgium", "Canada", "Denmark", "Finland", "France", "Iceland", "Ireland", "Italy", 
"Japan"]
# skipped Luxembourg, Netherlands, New Zealand, Norway, Spain, Sweden, Switzerland, United Kingdom and United States of America

# 111 * (i - 1) + 1, 111*i i denotes the time periods. 


function get_row(fullpath::String, t::Int) 
    # t : time period
    df = open(fullpath) do io
    readline(io)  # ignore metadata line
    CSV.read(io, DataFrame;
             delim=' ',
             ignorerepeated=true)
    end

    dx = df[(111*(t-1) + 1):(t * 111), "dx"]
    row = Vector{Float64}(undef, 100000)
    @assert sum(dx) == 100000
    start = 1
    for i in 1:length(dx)
        row[start:(start + dx[i])] .= fill(Float64(i - 1), dx[i])
        start += dx[i]
    end
    return row
end

function get_matrix(group::Vector{String}, t::Int, gender::String)
    # t : time period
    filepath = "mortality_dataset/group1/$(gender)"
    hier_sample_1 = Matrix{Float64}(undef, length(group), 100000)
    for i in 1:length(group)
        fullpath = joinpath(filepath,group[i]*"_"*gender*".txt")
        hier_sample_1[i,:] = get_row(fullpath, t)
    end
    return hier_sample_1
end



hier_sample_1 = get_matrix(group1, 2, "males")
#hier_sample_2 = get_matrix(group2)



# 111 length
# fullpath = "mortality_dataset/group1/males/belarus_males.txt"

# df = open(fullpath) do io
#     readline(io)  # ignore metadata line
#     CSV.read(io, DataFrame;
#              delim=' ',
#              ignorerepeated=true)
#     end



# sum(df[1:111,"dx"])