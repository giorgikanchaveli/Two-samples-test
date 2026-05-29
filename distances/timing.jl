# Run from project root: julia distances/timing.jl
#
# For each pair of laws of RPMs, measures wall-clock time for one call to
# dlip (HIPM) and ww (WoW) at fixed n and m.
# Reports mean elapsed seconds over N_REPS repetitions (after one warmup call).
#
# Case 1 — Dirichlet Process:  Q_1 = DP(1.0, Normal(0,1)),     Q_2 = DP(2.0, Normal(0,1))
# Case 2 — Normal-Normal:      Q_1 = normal_normal_A(0.0, 1.0), Q_2 = normal_normal_A(1.0, 1.0)

include(joinpath(pwd(), "methods.jl"))

const n      = 100
const m      = 500
const N_REPS = 7

function time_methods(q_1::LawRPM, q_2::LawRPM, n::Int, m::Int, n_reps::Int)
    # warmup: one call to each method to avoid measuring JIT compilation
    h_1 = generate_hiersample(q_1, n, m)
    h_2 = generate_hiersample(q_2, n, m)
    a = min(h_1.a, h_2.a)
    b = max(h_1.b, h_2.b)
    dlip(h_1, h_2, a, b)
    ww(h_1, h_2)

    t_hipm = 0.0
    t_wow  = 0.0
    for _ in 1:n_reps
        h_1 = generate_hiersample(q_1, n, m)
        h_2 = generate_hiersample(q_2, n, m)
        a = min(h_1.a, h_2.a)
        b = max(h_1.b, h_2.b)
        t_hipm += @elapsed dlip(h_1, h_2, a, b)
        t_wow  += @elapsed ww(h_1, h_2)
    end
    return t_hipm / n_reps, t_wow / n_reps
end

cases = [
    ("Dirichlet Process",
     DP(1.0, Normal(0.0, 1.0)),
     DP(2.0, Normal(0.0, 1.0)))
]

println("n = $n,  m = $m,  N_REPS = $N_REPS\n")
println(rpad("Case", 22), rpad("HIPM (s)", 14), "WoW (s)")
println("-"^50)
for (label, q_1, q_2) in cases
    t_hipm, t_wow = time_methods(q_1, q_2, n, m, N_REPS)
    println(rpad(label, 22),
            rpad(string(round(t_hipm, digits = 4)), 14),
            round(t_wow, digits = 4))
end
