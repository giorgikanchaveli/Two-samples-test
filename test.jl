# optimize wow code


include("distributions.jl")
include("structures.jl")
include("distances/distance_Wasserstein.jl")
include("distances/new_distance.jl")




function f_old(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)
    rej_rate = 0.0

    for s in 1:S
        hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        a = minimum([hier_sample_1.a, hier_sample_2.a])
        b = maximum([hier_sample_1.b, hier_sample_2.b])
        hier_sample_1.a = a
        hier_sample_2.a = a
        hier_sample_1.b = b
        hier_sample_2.b = b
        observed_test_stat = dlip(hier_sample_1, hier_sample_2)
        
        # obtain quantile using bootstrap approach
        boostrap_samples = zeros(n_boostrap) # zeros can be improved
        
        total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
        for i in 1:n_boostrap
            indices_1 = sample(1:2*n, n; replace = true)
            indices_2 = sample(1:2*n, n; replace = true)
            atoms_1 = total_rows[indices_1,:]  # resample from pooled hierarchical sample
            atoms_2 = total_rows[indices_2,:]  # resample from pooled hierarchical sample
            
        
            hier_sample_1_boostrap = emp_ppm(atoms_1, n, m, a, b)
            hier_sample_2_boostrap = emp_ppm(atoms_2, n, m, a, b)

            boostrap_samples[i] = dlip(hier_sample_1_boostrap, hier_sample_2_boostrap)
        end
        threshold = quantile(boostrap_samples, 1 - θ)
        
        rej_rate += 1.0*(observed_test_stat > threshold)
    end
    return rej_rate / S
end





function f_new(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)
    rej_rate = 0.0

    for s in 1:S
        hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        a = minimum((hier_sample_1.a, hier_sample_2.a))
        b = maximum((hier_sample_1.b, hier_sample_2.b))
        hier_sample_1.a = a
        hier_sample_2.a = a
        hier_sample_1.b = b
        hier_sample_2.b = b
        observed_test_stat = dlip(hier_sample_1, hier_sample_2)
        
        # obtain quantile using bootstrap approach
        boostrap_samples = zeros(n_boostrap) # zeros can be improved
        
        total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
        for i in 1:n_boostrap
            indices_1 = sample(1:2*n, n; replace = true)
            indices_2 = sample(1:2*n, n; replace = true)
            atoms_1 = total_rows[indices_1,:]  # resample from pooled hierarchical sample
            atoms_2 = total_rows[indices_2,:]  # resample from pooled hierarchical sample
            
        
            hier_sample_1_boostrap = emp_ppm(atoms_1, n, m, a, b)
            hier_sample_2_boostrap = emp_ppm(atoms_2, n, m, a, b)

            boostrap_samples[i] = dlip(hier_sample_1_boostrap, hier_sample_2_boostrap)
        end
        threshold = quantile(boostrap_samples, 1 - θ)
        
        rej_rate += 1.0*(observed_test_stat > threshold)
    end
    return rej_rate / S
end






# parameters for functions

n = 100
m = 100

a = 0.0
b = 1.0

p = 1

S = 1
n_boostrap = 1
θ = 0.05


q_1 = tnormal_normal(1.0,1.0,-10.0,10.0)
q_2 = tnormal_normal(1.0,1.0,-10.0,10.0)


# check that values math
Random.seed!(12345)
old_value = f_old(q_1,q_2,n,m,S,θ,n_boostrap)
Random.seed!(12345)
new_value = f_new(q_1,q_2,n,m,S,θ,n_boostrap)
@assert abs(old_value - new_value) < 1e-8

# compare times
Random.seed!(12345)
time_old = @elapsed f_old(q_1,q_2,n,m,S,θ,n_boostrap)
Random.seed!(12345)
time_new = @elapsed f_new(q_1,q_2,n,m,S,θ,n_boostrap)


println("improvement difference : $(time_old - time_new)")
println("improvement ratio : $(time_old / time_new)")