using CSV, DataFrames

group1 = ["belarus", "Bulgaria", "Czechia", "Estonia", "Hungary", "Latvia", "Poland", "Lithuania", "Russia", "Slovakia", "Ukraine"]




function get_row(fullpath::String)
    df = open(fullpath) do io
    readline(io)  # ignore metadata line
    CSV.read(io, DataFrame;
             delim=' ',
             ignorerepeated=true)
    end
    dx = Float64.(df[1:111, "dx"])
    row = Vector{Float64}(undef, 100000)

    cums = cumsum(dx)
    for i in 1:length(dx)
        start = (i-1)*cums
        row[((i-1)*Int(dx[i]) + 1): i*Int(dx[i])] .= fill(Float64(i - 1), Int(dx[i]))
    end
    return row
end

function get_matrix(group::Vector{String})
    filepath = "mortality_dataset/group1/males"
    hier_sample_1 = Matrix{Float64}(undef, length(group), 100000)
    for i in 1:length(group)
        fullpath = joinpath(filepath,group[i]*"_males.txt")
        hier_sample_1[i,:] = get_row(fullpath)
    end
    return hier_sample_1
end



hier_sample_1 = get_matrix(group1)

# 111 length
