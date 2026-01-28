
include("methods.jl")
using Plots


n_1 = 70
n_2 = 70
n = n_1
m = 100

# θ = collect(0.0:0.01:1.0)
θ = 0.05
n_samples = 3
bootstrap = true


S = 3
q_1 = tnormal_normal(1.0, 1.0, -10.0, 10.0)
q_2 = mixture(q_1, tnormal_normal(1.0, 1.0, -10.0, 10.0), 0.6)

@time r = rejection_rate_all(q_1, q_2, n, m, S, θ, n_samples, bootstrap)





function test_statistic_energy_1(atoms_1::AbstractArray{Float64, 2}, atoms_2::AbstractArray{Float64, 2})
    
    n = size(atoms_1)[1]
    n == size(atoms_2)[1] || throw(ArgumentError("Number of rows of atoms_1 and atoms_2 are not the same."))
    
    sum_distances_x = 0.0 # collects sum of all possible distances in atoms_1
    sum_distances_xy = 0.0 # collects sum of all possible distances between atoms_1 and atoms_2
    sum_distances_y = 0.0 # collects sum of all possible distances in atoms_2

    for i in 1:n
        x = @view atoms_1[i,:]
        y = @view atoms_2[i,:]
        for j in 1:n
            x_j = @view atoms_1[j,:]
            y_j = @view atoms_2[j,:]

            sum_distances_x += wasserstein_1d_equal(x, x_j)
            sum_distances_xy += wasserstein_1d_equal(x, y_j)
            sum_distances_y += wasserstein_1d_equal(y, y_j)
        end
    end
    distance = 2 * sum_distances_xy / (n * n) - sum_distances_x / (n * n) -sum_distances_y / (n * n)
    return distance * n / 2
end
