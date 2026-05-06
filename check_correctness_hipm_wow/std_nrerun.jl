using Statistics
using Plots
using FLoops

include("../distances/hipm.jl")
include("../distributions.jl")


function std_per_nrerun(k::Int)
    """
    Estimate standard deviation of hipm where we consider randomness only from computation of hipm.
    """
    n_rerun_values = [5, 10, 20, 50, 100]
    S = 50  # Number of trials per n_rerun

    # Store std devs for each sample
    all_std_devs = [Dict() for _ in 1:k]

    println("Estimating standard deviations using $k different hierarchical samples...")
    println("Each sample tested with S=$S trials per n_rerun\n")

    @floop ThreadedEx() for sample_idx in 1:k
        # println("Sample $sample_idx:")
        # Generate one pair of hierarchical samples
        h_1 = generate_hiersample(normal_normal_B(1.0, 1.0, 2.0), 100, 100)
        h_2 = generate_hiersample(normal_normal_B(1.0, 1.0, 2.0), 100, 100)

        a = min(h_1.a, h_2.a)
        b = max(h_1.b, h_2.b)

        # For each n_rerun store standard deviation 
        for n_rerun in n_rerun_values
            outputs = zeros(S)

            for trial in 1:S
                outputs[trial] = dlip(h_1, h_2, a, b; n_rerun=n_rerun)
            end

            std_value = std(outputs)
            all_std_devs[sample_idx][n_rerun] = std_value
            # println("  n_rerun=$n_rerun: std dev = $(round(std_value, sigdigits=6))")
        end
        # println()
    end

    println("="^50)
    println("Average Standard Deviations across all $k samples:")
    println("n_rerun\tAvg Std Dev")
    println("-------\t-----------")

    for n_rerun in n_rerun_values
        avg_std = mean([all_std_devs[i][n_rerun] for i in 1:k])
        println("$n_rerun\t$(round(avg_std, sigdigits=6))")
    end

    return all_std_devs
end

println("Running with $(Threads.nthreads()) threads")
Random.seed!(2234)
elapsed = @elapsed all_std_devs = std_per_nrerun(50)
println("done")
println("Total computation time: $(round(elapsed, digits=2)) seconds")

# Plot std devs per n_rerun
n_rerun_values = sort(collect(keys(all_std_devs[1])))
k = length(all_std_devs)

avg_stds = [mean([all_std_devs[i][n] for i in 1:k]) for n in n_rerun_values]

p = plot(n_rerun_values, log.(avg_stds),
    label="Average std dev",
    lw=2, marker=:circle, markersize=6,
    color=:black,
    xlabel="n_rerun", ylabel="log (Std Dev)",
    title="Std Dev of dlip per n_rerun",
    legend=:topright)


savefig(p, joinpath(@__DIR__, "nrerun_std_plot.png"))
println("Plot saved")