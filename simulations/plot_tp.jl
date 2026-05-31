using Plots
using ArgParse
using ExtractMacro

include(joinpath(pwd(),"methods.jl"))

rpms_1 = [beta_beta_A(1.0, 1.0),
        normal_normal_A(0.0, 1.0),
        DP(1.0, Uniform(0,1)),
        DP(1.0, Normal(0.0, 1.0))
       ]

rpms_2 = [beta_beta_A(1.0, 1.5),
          normal_normal_A(0.0, 1.5),
          DP(2.0, Uniform(0.0, 1.0)),
          DP(1.0, Beta(1.0,1.4))
          ]

function parse_commandline()
    s = ArgParseSettings(description = "Run TP Simulations and Plot")
    @add_arg_table! s begin
        "--label_q_1"
            help = "label for first law of RPM"
            arg_type = Int
            default = 2
        "--label_q_2"
            help = "label for second law of RPM"
            arg_type = Int
            default = 2
        "--n"
            help = "number of rows n"
            arg_type = Int
            default = 100
        "--m"
            help = "number of columns m"
            arg_type = Int
            default = 100
        "--S"
            help = "number of mcmc iterations"
            arg_type = Int
            default = 16
        "--n_perm"
            help = "number of samples via permutation approach"
            arg_type = Int
            default = 70
    end
    return parse_args(s)
end

struct SimulationConfig
    Q_1::LawRPM
    Q_2::LawRPM
    n::Int
    m::Int
    S::Int
    n_perm::Int
end

function run_simulation(config::SimulationConfig)
    @extract config : Q_1 Q_2 n m S n_perm
    θs = collect(0.0:0.01:1.0)
    tp_hipm, tp_wow = rejection_rate_hipm_wow(Q_1, Q_2, n, m, S, θs, n_perm, false)
    return θs, tp_hipm, tp_wow
end

function main()
    parsed_args = parse_commandline()
    label_q_1 = parsed_args["label_q_1"]
    label_q_2 = parsed_args["label_q_2"]
    Q_1 = rpms_1[label_q_1]
    Q_2 = rpms_2[label_q_2]
    n, m, S, n_perm = parsed_args["n"], parsed_args["m"], parsed_args["S"], parsed_args["n_perm"]
    config = SimulationConfig(Q_1, Q_2, n, m, S, n_perm)
    θs, tp_hipm, tp_wow = run_simulation(config)

    title_str = "Power"

    p = plot(
        θs, tp_hipm,
        label = "HIPM",
        linewidth = 2,
        marker = :circle,
        xlabel = "θ",
        ylabel = "Power",
        title = title_str,
        legend = :best,
        aspect_ratio = :equal,
        xlims = (0, 1),
        ylims = (0, 1)
    )
    plot!(p, θs, tp_wow,
        label = "WoW",
        linewidth = 2,
        marker = :square
    )

    display(p)
end

@info "number of threads: $(Threads.nthreads())"
t_start = time()
main()
@info "Total duration: $(round(time() - t_start, digits=2)) seconds"
