include("methods.jl")

using Plots
using DataFrames



function sample_distances(Q_1::LawRPM, Q_2::LawRPM, n::Int, m::Int, nReps::Int)
    # for given Q_1,Q_2 laws of RPM, compute empirical distance between them nReps times. 
    distances_wow = zeros(nReps)
    distances_hipm = zeros(nReps)

    @floop ThreadedEx() for i in 1:nReps
        h_s_1, h_s_2 = generate_hiersample(Q_1, n, m), generate_hiersample(Q_2, n, m)
        distances_wow[i] = ww(h_s_1, h_s_2)

        a = minimum((h_s_1.a, h_s_2.a))
        b = maximum((h_s_1.b, h_s_2.b))

        distances_hipm[i] = dlip(h_s_1, h_s_2, a,b)

    end
    return distances_wow, distances_hipm
end




function values_for_fig_left()
    Q_1 = DP(1.0, Uniform(-0.5, 0.5))
    Q_2 = DP(1.0, MixtureModel([Uniform(-1.0, -0.75), Uniform(0.75, 1.0)]))

    nReps = 24
    ns = [16, 32, 64, 128, 256, 512]
    m = 5000
    
    mean_distances_wow = zeros(length(ns))
    mean_distances_hipm = zeros(length(ns))

    std_distances_wow = zeros(length(ns))
    std_distances_hipm = zeros(length(ns))

    for (i,n) in enumerate(ns)
        distances_wow, distances_hipm = sample_distances(Q_1, Q_2, n, m, nReps)

        mean_distances_wow[i] = mean(distances_wow)
        std_distances_wow[i] = std(distances_wow)

        mean_distances_hipm[i] = mean(distances_hipm)
        std_distances_hipm[i] = std(distances_hipm)
    end

    return DataFrame(
        ns = ns,
        mean_distances_wow = mean_distances_wow,
        mean_distances_hipm = mean_distances_hipm,
        std_distances_wow = std_distances_wow,
        std_distances_hipm = std_distances_hipm
        )
end



function values_for_fig_right()
    Q = DP(1.0, Uniform(0.0, 1.0))

    nReps = 24
    ns = [16, 32, 64, 128, 256, 512]
    m = 5000
    
    mean_distances_wow = zeros(length(ns))
    mean_distances_hipm = zeros(length(ns))

    std_distances_wow = zeros(length(ns))
    std_distances_hipm = zeros(length(ns))

    for (i,n) in enumerate(ns)
        distances_wow, distances_hipm = sample_distances(Q, Q, n, m, nReps)

        mean_distances_wow[i] = mean(distances_wow)
        std_distances_wow[i] = std(distances_wow)

        mean_distances_hipm[i] = mean(distances_hipm)
        std_distances_hipm[i] = std(distances_hipm)
    end

    return DataFrame(
        ns = ns,
        mean_distances_wow = mean_distances_wow,
        mean_distances_hipm = mean_distances_hipm,
        std_distances_wow = std_distances_wow,
        std_distances_hipm = std_distances_hipm
        )
end

function plot_fig_left(df::DataFrame)
    nReps = 24
    plot_title = "Fig 1, Left"

    hipm_band = 1.96 .* df.std_distances_hipm ./ sqrt(nReps)
    wow_band  = 1.96 .* df.std_distances_wow  ./ sqrt(nReps)

    p = plot(
        df.ns, df.mean_distances_hipm;
        ribbon = hipm_band,
        label = "HIPM",
        xlabel = "n",
        ylabel = "mean distance",
        title = plot_title,
        linewidth = 2,
        marker = :circle,
        markersize = 4,
        xticks = 0:100:500,
        yticks = 0.4:0.1:0.7,
        xlims = (0, 500),     
        ylims = (0.37, 0.7)    
    )

    plot!(
        p,
        df.ns, df.mean_distances_wow;
        ribbon = wow_band,
        label = "WOW",
        linewidth = 2,
        marker = :square,
        markersize = 4
    )

    hline!(
        p,
        [5/8],
        linestyle = :dash,
        linewidth = 2,
        color = :black,
        label = "true value"
    )

    return p
end


function plot_fig_right(df::DataFrame)
    nReps = 24
    plot_title = "Fig 1, Right"

    hipm_band = 1.96 .* df.std_distances_hipm ./ sqrt(nReps)
    wow_band  = 1.96 .* df.std_distances_wow  ./ sqrt(nReps)

    p = plot(
        df.ns,
        df.mean_distances_hipm;
        ribbon = hipm_band,
        label = "HIPM",
        xlabel = "n",
        ylabel = "mean distance",
        title = plot_title,
        linewidth = 2,
        marker = :circle,
        markersize = 4,
        xscale = :log10,
        yscale = :log10,
        
        xticks = [10^1.5, 100, 10^(2.5)],
        yticks = [10^(-2), 10^(-1)]
    )

    plot!(
        p,
        df.ns,
        df.mean_distances_wow;
        ribbon = wow_band,
        label = "WOW",
        linewidth = 2,
        marker = :square,
        markersize = 4
    )


    return p
end


df_fig_left = values_for_fig_left()
df_fig_right = values_for_fig_right()
plot_left = plot_fig_left(df_fig_left)
plot_right = plot_fig_right(df_fig_right)




