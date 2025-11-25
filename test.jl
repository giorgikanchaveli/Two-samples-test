# optimize wow code


include("distributions.jl")
include("structures.jl")
include("distances/distance_Wasserstein.jl")
include("distances/new_distance.jl")
using RCall # to call R functions




    
function f_old(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)

    rej_rate = 0.0
    
 
    for s in 1:S
     
        hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        atoms_1, atoms_2 = hier_sample_1.atoms, hier_sample_2.atoms

        @rput atoms_1 atoms_2 n n_boostrap
        R"""
        # if (!requireNamespace("frechet", quietly = TRUE)) {
        #   install.packages("frechet", repos="https://cloud.r-project.org")
        # }
        library(frechet)
        atoms_all = rbind(atoms_1, atoms_2)

        group <- c(rep(1, n), rep(2, n))

        result_denanova = DenANOVA(
            yin = atoms_all,
            group = group,
            optns = list(boot = TRUE, R = n_boostrap)
        )
        pvalue = result_denanova$pvalBoot
        """
        @rget pvalue
        rej_rate += 1 * (pvalue < θ)
    end
    rej_rate /= S
    return rej_rate
end



function f_new(q_1::PPM, q_2::PPM, n::Int, m::Int, S::Int, θ::Float64, n_boostrap::Int)

    rej_rate = 0.0
    
 
    for s in 1:S
     
        hier_sample_1, hier_sample_2 = generate_emp(q_1, n, m), generate_emp(q_2, n, m)
        atoms_1, atoms_2 = hier_sample_1.atoms, hier_sample_2.atoms

        @rput atoms_1 atoms_2 n n_boostrap
        R"""
        # if (!requireNamespace("frechet", quietly = TRUE)) {
        #   install.packages("frechet", repos="https://cloud.r-project.org")
        # }
        library(frechet)
        atoms_all = rbind(atoms_1, atoms_2)

        group <- c(rep(1, n), rep(2, n))

        result_denanova = DenANOVA(
            yin = atoms_all,
            group = group,
            optns = list(boot = TRUE, R = n_boostrap)
        )
        pvalue = result_denanova$pvalBoot
        """
        @rget pvalue
        rej_rate += 1 * (pvalue < θ)
    end
    rej_rate /= S
    return rej_rate
end











# parameters for functions

n = 100
m = 100

a = 0.0
b = 1.0

p = 1

S = 1
n_boostrap = 100
θ = 0.05


q_1 = tnormal_normal(1.0,1.0,-10.0,10.0)
q_2 = tnormal_normal(1.0,1.0,-10.0,10.0)

hier_sample_1 = generate_emp(q_1, n, m)
hier_sample_2 = generate_emp(q_2, n, m)




# check that values math
Random.seed!(12345)
old_value = f_old(q_1, q_2, n, m, S, θ, n_boostrap)
Random.seed!(12345)
new_value = f_new(q_1, q_2, n, m, S, θ, n_boostrap)
@assert abs(old_value - new_value) < 1e-8

# compare times
Random.seed!(12345)
time_old = @elapsed f_old(q_1, q_2, n, m, S, θ, n_boostrap)
Random.seed!(12345)
time_new = @elapsed f_new(q_1, q_2, n, m, S, θ, n_boostrap)


println("improvement difference : $(time_old - time_new)")
println("improvement ratio : $(time_old / time_new)")
println("new time : $(time_new)")


