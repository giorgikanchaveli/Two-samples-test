include("distributions.jl")

tn_n = tnormal_normal(1.0, 2.0, -10.0, 10.0)
dp = DP(0.001, Beta(1,1))
d_n = discr_normal([-1.0, 1.0, 2.0], [0.2, 0.0, 0.8])



n, m = 1000, 10
hs_tn_n = generate_hiersample(tn_n, n, m)
hs_dp = generate_hiersample(dp, n, m)
hs_d_n = generate_hiersample(d_n, n, m)


count = [0, 0]
for i in 1:n
    if abs(hs_d_n.atoms[i, 1] + 1.0) < 1e-2
        count[1] += 1
    end
    if abs(hs_d_n.atoms[i, 1] - 2.0) < 1e-2
        count[2] += 1
    end
end