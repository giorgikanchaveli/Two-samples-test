using Plots
using ArgParse
using ExtractMacro

include(joinpath(pwd(),"methods.jl"))

rpms = [beta_beta_A(1.0, 1.0),
        normal_normal_A(0.0, 1.0),
        DP(1.0, Uniform(0,1)),
        DP(1.0, Normal(0.0, 1.0))
        ]


function parse_commandline()
    s = ArgParseSettings(description = "Run FP Simulations and Plot")
    @add_arg_table! s begin
        "--label_q"
            help = "label for law of RPM"
            arg_type = Int
            default = 1
        "--n"
            help = "number of rows n"
            arg_type = Int
            default = 1
        "--m"
            help = "number of columns m"
            arg_type = Int
            default = 1
        "--S"
            help = "number of mcmc iterations"
            arg_type = Int
            default = 2
        "--n_perm"
            help = "number of samples via permutation approach"
            arg_type = Int
            default = 2
    end
    return parse_args(s)
end

struct SimulationConfig
    Q::LawRPM
    n::Int
    m::Int
    S::Int
    n_perm::Int
end

function run_simulation(config::SimulationConfig)
    @extract config : Q n m S n_perm
    θs = collect(0.0:0.01:1.0)
    fp_hipm, fp_wow = rejection_rate_hipm_wow(Q, Q, n, m, S, θs, n_perm, false)
    return θs, fp_hipm, fp_wow
end

function main()
    parsed_args = parse_commandline()
    label_q = parsed_args["label_q"]
    Q = rpms[label_q]
    n, m, S, n_perm = parsed_args["n"], parsed_args["m"], parsed_args["S"], parsed_args["n_perm"]
    config = SimulationConfig(Q, n, m, S, n_perm)
    θs, fp_hipm, fp_wow = run_simulation(config)

    title_str = "False Positive Rates"

    p = plot(
        θs, fp_hipm,
        label = "hipm",
        linewidth = 2,
        marker = :circle,
        xlabel = "θ",
        ylabel = "False Positive Rate",
        title = title_str,
        legend = :best,
        aspect_ratio = :equal,
        xlims = (0, 1),
        ylims = (0, 1)
    )
    plot!(p, θs, fp_wow,
        label = "wow",
        linewidth = 2,
        marker = :square
    )
    plot!(p, [0, 1], [0, 1],
        label = "",
        linewidth = 1,
        linestyle = :dash,
        color = :black
    )

    display(p)
end

@info "number of threads: $(Threads.nthreads())"
t_start = time()
main()
@info "Total duration: $(round(time() - t_start, digits=2)) seconds"
