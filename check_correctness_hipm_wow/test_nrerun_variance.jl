using Statistics
using Plots
include("../distances/hipm.jl")
include("../distributions.jl")

function test_nrerun_variance()
    # Generate one pair of hierarchical samples
    println("Generating hierarchical samples from normal_normal_B...")
    h_1 = generate_hiersample(normal_normal_B(1.0, 2.0, 3.0), 10, 100)
    h_2 = generate_hiersample(normal_normal_B(1.0, 2.0, 3.0), 10, 100)

    a = min(minimum(h_1.atoms), minimum(h_2.atoms))
    b = max(maximum(h_1.atoms), maximum(h_2.atoms))

    # Test different n_rerun values
    n_rerun_values = [5, 10, 20, 50, 100]
    S = 10  # Number of trials per n_rerun

    println("\nComputing standard deviations for different n_rerun values...")
    println("Each n_rerun will be tested S=$S times on the same input\n")
    println("n_rerun\tStd Dev")
    println("-------\t-------")

    std_devs = Dict()

    for n_rerun in n_rerun_values
        outputs = zeros(S)

        for trial in 1:S
            outputs[trial] = dlip(h_1, h_2, a, b; n_rerun=n_rerun)
        end

        std_value = std(outputs)
        std_devs[n_rerun] = std_value

        println("$n_rerun\t$(round(std_value, sigdigits=6))")
    end

    println("\n" * "="^50)
    println("Summary:")
    for n_rerun in n_rerun_values
        println("  n_rerun=$n_rerun: std dev = $(std_devs[n_rerun])")
    end

    return std_devs
end

function test_nrerun_variance_multiple(k::Int)
    """
    Test n_rerun on k different hierarchical samples and average the results.
    """
    n_rerun_values = [5, 10, 20, 50, 100]
    S = 10  # Number of trials per n_rerun

    # Store std devs for each sample
    all_std_devs = [Dict() for _ in 1:k]

    println("Testing on $k different hierarchical samples...")
    println("Each sample tested with S=$S trials per n_rerun\n")

    for sample_idx in 1:k
        println("Sample $sample_idx:")
        # Generate one pair of hierarchical samples
        h_1 = generate_hiersample(normal_normal_B(1.0, 2.0, 2.0), 100, 100)
        h_2 = generate_hiersample(normal_normal_B(1.0, 1.0, 1.5), 100, 100)

        a = min(minimum(h_1.atoms), minimum(h_2.atoms))
        b = max(maximum(h_1.atoms), maximum(h_2.atoms))

        for n_rerun in n_rerun_values
            outputs = zeros(S)

            for trial in 1:S
                outputs[trial] = dlip(h_1, h_2, a, b; n_rerun=n_rerun)
            end

            std_value = std(outputs)
            all_std_devs[sample_idx][n_rerun] = std_value
            println("  n_rerun=$n_rerun: std dev = $(round(std_value, sigdigits=6))")
        end
        println()
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

# Run single sample test
# println("="^50)
# println("SINGLE SAMPLE TEST")
# println("="^50)
# test_nrerun_variance()

# Run multiple samples test (k=3)
println("\n\n" * "="^50)
println("MULTIPLE SAMPLES TEST (k=3)")
println("="^50)
all_std_devs = test_nrerun_variance_multiple(10)
println("done")

# Plot std devs per n_rerun
n_rerun_values = sort(collect(keys(all_std_devs[1])))
k = length(all_std_devs)

avg_stds = [mean([all_std_devs[i][n] for i in 1:k]) for n in n_rerun_values]

p = plot(n_rerun_values, avg_stds,
    label="Average std dev",
    lw=2, marker=:circle, markersize=6,
    color=:black,
    xlabel="n_rerun", ylabel="Std Dev",
    title="Std Dev of dlip vs n_rerun",
    legend=:topright)

for i in 1:k
    sample_stds = [all_std_devs[i][n] for n in n_rerun_values]
    plot!(p, n_rerun_values, sample_stds,
        label="Sample $i", lw=1, alpha=0.4, marker=:circle, markersize=3)
end

savefig(p, joinpath(@__DIR__, "nrerun_variance_plot.png"))
println("Plot saved to nrerun_variance_plot.png")