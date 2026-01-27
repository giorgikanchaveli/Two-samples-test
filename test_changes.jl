include("distributions.jl")
include("distances/wow.jl")
include("distances/hipm.jl")
include("distances/hipm_old.jl")



m = 50
n_1 = 50
n_2 = 50

h_1 = rand(n_1, m)
h_2 = rand(n_2, m)
weights_1 = fill(1.0/m, (n_1, m))
weights_2 = fill(1.0/m, (n_2, m))


a = minimum((minimum(h_1),minimum(h_2)))
b = maximum((maximum(h_1),maximum(h_2)))

Random.seed!(1234)
d_old = dlip_old_diffsize(h_1, h_2, weights_1, weights_2, a, b; maxTime = 5.0)
Random.seed!(1234)
d_new = dlip(h_1, h_2, weights_1, weights_2, a, b; max_time = 5.0)



distance_w1d = mean(abs.(sort(h_1[:,1]) .- sort(h_2[:,1])))







distance_ww = ww(h_1, h_2)
distance_hipm = dlip(h_1, h_2, 0.0,1.0)

println("true distance is : $(distance_w1d)")
println("ww is : $(distance_ww)")
println("hipm is : $(distance_hipm)")

println("$(abs(distance_w1d-distance_hipm))")
println("$(abs(distance_w1d-distance_ww))")



