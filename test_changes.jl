
include("distances/hipm.jl")
include("distances/hipm_old.jl")
include("distributions.jl")


n_1 = 50
n_2 = 50
m = 1


h_1, h_2 = generate_hiersample(DP(1.0, Beta(1,2)), n_1, m), generate_hiersample(DP(3.0, Beta(1,4)), n_2, m)
weights_1, weights_2 = rand(n_1, m), rand(n_2, m)

for i in 1:n_1
    weights_1[i,:] .= weights_1[i,:] ./ sum(weights_1[i,:])
end
for i in 1:n_2
    weights_2[i,:] .= weights_2[i,:] ./ sum(weights_2[i,:])
end



d_dlip = dlip(h_1.atoms, h_2.atoms, weights_1, weights_2, 0.0, 1.0; max_time = 1.0)
d_dlip_old = dlip_old_diffsize(h_1.atoms, h_2.atoms, weights_1, weights_2, 0.0,1.0; maxTime = 1.0)


@assert abs(d_dlip - d_dlip_old) < 1e-2 "new dlip and old dlip are not the same"

println("dlip is $(d_dlip)")
println("original is $(d_dlip_old)")








# weights_1 = fill(1.0/m, (n_1, m))
# weights_2 = fill(1.0/m, (n_2, m))


# a = minimum((minimum(h_1),minimum(h_2)))
# b = maximum((maximum(h_1),maximum(h_2)))

# Random.seed!(1234)
# d_old = dlip_old_diffsize(h_1, h_2, weights_1, weights_2, a, b; maxTime = 5.0)
# Random.seed!(1234)
# d_new = dlip(h_1, h_2, weights_1, weights_2, a, b; max_time = 5.0)

 distance_w1d = mean(abs.(sort(h_1.atoms[:,1]) .- sort(h_2.atoms[:,1])))

 distance_ww = ww(h_1, h_2)
# distance_hipm = dlip(h_1, h_2, 0.0,1.0)

# println("true distance is : $(distance_w1d)")
# println("ww is : $(distance_ww)")
# println("hipm is : $(distance_hipm)")

# println("$(abs(distance_w1d-distance_hipm))")
# println("$(abs(distance_w1d-distance_ww))")



