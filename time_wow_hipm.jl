include("distributions.jl")
include("distances/wow.jl")
include("distances/hipm.jl")

using BenchmarkTools

using FLoops

# I want to obtain the number of seconds needed to finish simulations for one pair.
# I simulate S samples and obtain threshold per each samples. I check times for several
# threads.

# n_threads = [1,4,12,20,30,40]

function threshold_wow_nothread(hier_sample_1::HierSample, hier_sample_2::HierSample, θ::Float64, n_samples::Int, bootstrap::Bool)
    n = hier_sample_1.n
    atoms_1 = hier_sample_1.atoms
    atoms_2 = hier_sample_2.atoms
    
    samples = zeros(n_samples)
    total_rows = vcat(atoms_1, atoms_2) # collect all rows
    if bootstrap
        for i in 1:n_samples
            indices_1 = sample(1:2*n, n; replace = true)
            indices_2 = sample(1:2*n, n; replace = true)

            new_atoms_1 = total_rows[indices_1,:] # first rows indexed by n random indices to the atoms_1
            new_atoms_2 = total_rows[indices_2,:] # first rows indexed by n random indices to the atoms_2

            samples[i] = ww(new_atoms_1, new_atoms_2) # sorted = true
        end
    else
        for i in 1:n_samples
            random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

            new_atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
            new_atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
         
            samples[i] = ww(new_atoms_1, new_atoms_2) # sorted = true
        end
    end
    return quantile(samples, 1 - θ)
end



function threshold_hipm_nothread(hier_sample_1::HierSample, hier_sample_2::HierSample, θ::Float64, n_samples::Int, bootstrap::Bool)
    n = hier_sample_1.n
    m = hier_sample_1.m
    a = minimum((hier_sample_1.a,hier_sample_2.a))
    b = maximum((hier_sample_1.b,hier_sample_2.b))
   
    samples = zeros(n_samples)
    total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
    if bootstrap
        for i in 1:n_samples
            indices_1 = sample(1:2*n, n; replace = true)
            indices_2 = sample(1:2*n, n; replace = true)

            new_atoms_1 = total_rows[indices_1,:] # first rows indexed by n random indices to the atoms_1
            new_atoms_2 = total_rows[indices_2,:] # first rows indexed by n random indices to the atoms_2

            new_h_1 = HierSample(new_atoms_1, a, b)
            new_h_2 = HierSample(new_atoms_2, a, b)

            samples[i] = dlip(new_h_1, new_h_2, a, b)
        end
    else
        for i in 1:n_samples
            random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

            new_atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
            new_atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
            new_h_1 = HierSample(new_atoms_1, a, b)
            new_h_2 = HierSample(new_atoms_2, a, b)

            samples[i] = dlip(new_h_1, new_h_2, a, b)
        end
    end
    return quantile(samples, 1 - θ)
end


function threshold_hipm_new(hier_sample_1::HierSample, hier_sample_2::HierSample, θ::Float64, n_samples::Int, bootstrap::Bool)
    n = hier_sample_1.n
    m = hier_sample_1.m
    a = minimum((hier_sample_1.a,hier_sample_2.a))
    b = maximum((hier_sample_1.b,hier_sample_2.b))

   
    samples = zeros(n_samples)
    total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
    if bootstrap
        for i in 1:n_samples
            indices_1 = sample(1:2*n, n; replace = true)
            indices_2 = sample(1:2*n, n; replace = true)

            v1 = view(total_rows, indices_1, :)
            v2 = view(total_rows, indices_2, :)
            #new_atoms_1 = total_rows[indices_1,:] # first rows indexed by n random indices to the atoms_1
            #new_atoms_2 = total_rows[indices_2,:] # first rows indexed by n random indices to the atoms_2

            samples[i] = dlip(v1, v2, a, b)
            #samples[i] = dlip_new(new_atoms_1, new_atoms_2, a, b)
        end
    else
        for i in 1:n_samples
            random_indices = randperm(2*n) # indices to distribute rows to new hierarchical meausures

            v1 = view(total_rows, random_indices[1:n], :)
            v2 = view(total_rows, random_indices[n+1:end], :)
        
            #new_atoms_1 = total_rows[random_indices[1:n],:] # first rows indexed by n random indices to the atoms_1
            #new_atoms_2 = total_rows[random_indices[n+1:end],:] # first rows indexed by n random indices to the atoms_2
        
            samples[i] = dlip(v1, v2, a, b)
            #samples[i] = dlip_new(new_atoms_1, new_atoms_2, a, b)
            #samples[i] = ww(new_atoms_1, new_atoms_2)
        end
    end
    return quantile(samples, 1 - θ)
end





# q_1 = tnormal_normal(1.0,2.0,-10.0,10.0)
# q_2 = tnormal_normal(2.0,2.0,-10.0,10.0)
q_1 = DP(1.0, Beta(1,1), 0.0, 1.0)
q_2 = DP(13.0, Beta(1,1), 0.0, 1.0)
n = 100
m = 200



n_samples = 100

S = 1

θ = 0.05
bootstrap = false

h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)

#@btime threshold_wow_nothread(h_1, h_2, θ, n_samples, bootstrap)
@btime threshold_hipm_nothread(h_1, h_2, θ, n_samples, bootstrap)
@btime threshold_hipm_new(h_1, h_2, θ, n_samples, bootstrap)

a = minimum((h_1.a, h_2.a))
b = maximum((h_1.b, h_2.b))
@btime ww(h_1, h_2)
#@btime dlip(h_1, h_2, a, b)
# atoms_1 = h_1.atoms
# atoms_2 = h_2.atoms
@btime dlip(h_1, h_2, a, b)




# t= time()
# for i in 1:S
#     h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
#     threshold_wow_nothread(h_1, h_2, θ, n_samples, bootstrap)
# end
# dur_wass = time() - t

# # @floop ThreadedEx() 
# t= time()
# for i in 1:S
#     h_1, h_2 = generate_hiersample(q_1, n, m), generate_hiersample(q_2, n, m)
#     threshold_hipm_nothread(h_1, h_2, θ, n_samples, bootstrap)
# end
# dur_dlip = time() - t

# println("Bootstrap = $(bootstrap)")
# println("for threads = $(Threads.nthreads()), S = $S, n = $n, m = $m, n_samples = $(n_samples)")
# println("WoW : $(dur_wass) seconds ")
# println("HIPM : $(dur_dlip) seconds")