include("distributions.jl")
include("distances/distance_Wasserstein.jl")
include("distances/new_distance.jl")

using FLoops

# I want to obtain the number of seconds needed to compute distance (hipm, wow) between hierarchical
# samples 1000 times per each number of threads.

# n_threads = [1,4,12,20,30,40]

q_1 = tnormal_normal(1.0,2.0,-10.0,10.0)
q_2 = tnormal_normal(2.0,2.0,-10.0,10.0)
n = 100
m = 200

h_1, h_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
a = minimum((h_1.a,h_2.a))
b = maximum((h_1.b,h_2.b))


S = 1000


t= time()
@floop ThreadedEx() for i in 1:S
    ww(h_1, h_2)
end
dur_wass = time() - t


t= time()
@floop ThreadedEx() for i in 1:S
    dlip(h_1, h_2, a, b)
end
dur_dlip = time() - t

println("for threads = $(Threads.nthreads()), S = $S, n = $n, m = $m")
println("WoW : $(dur_wass) seconds ")
println("HIPM : $(dur_dlip) seconds")