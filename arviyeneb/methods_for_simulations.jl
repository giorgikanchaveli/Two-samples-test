using Plots
using KernelDensity
using RCall # to call R functions



include("distributions.jl")

include("distances/new_distance.jl")
include("distances/distance_Wasserstein.jl")
using DataFrames
using CSV


# On the first installation of RCall, set up the path for julia to find R. 
# ENV["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
# using Pkg
# Pkg.build("RCall")






# Methods for Dubey & Muller

function test_statistic_dm(μ_1::Vector{Float64}, μ_2::Vector{Float64})
    # This function computes the test statistic according to Dubey & Muller for a given two samples of means of Normal distributions
    # Inputs:
        # μ_1, μ_2 : two samples of means of Normal distributions, each of which is a vector of Float64 numbers
    
    n = length(μ_1)
    # Recall that we assume that each of the sample is Normal distribution, and for W_2 metric we only need their means.
    X = μ_1 # collect all the means from first samples of probability measures
    Y = μ_2 # collect all the means from second samples of probability measures

    μ_hat_1 = mean(X)
    μ_hat_2 = mean(Y)

    v_hat_1 = mean((X .- μ_hat_1).^2)
    v_hat_2 = mean((Y .- μ_hat_2).^2)

    σ_hat_1_squared = mean((X .- μ_hat_1).^4) - (v_hat_1)^2
    σ_hat_2_squared = mean((Y .- μ_hat_2).^4) - (v_hat_2)^2
    
    μ_hat_p = sum(X .+ Y) / (2 * n)
    v_hat_p = sum( (X .- μ_hat_p).^2 .+ (Y .- μ_hat_p).^2 ) / (2 * n)

    F_n = v_hat_p - v_hat_1/2 - v_hat_2/2
    U_n = (1/4) * ((v_hat_1 - v_hat_2)^2) / (σ_hat_1_squared * σ_hat_2_squared) 
    
    T_n = 2*n*U_n / (1/(2*σ_hat_1_squared) + 1/(2*σ_hat_2_squared) ) + 2 * n * (F_n^2) / (σ_hat_1_squared/4 + σ_hat_2_squared/4)

    return T_n
end

function decide_dm_asympt(pms_1::Vector{Float64}, pms_2::Vector{Float64}, n_boostrap::Int, θ = 0.05)
    # This function implements the two-sample test according to Dubey & Muller using asymptotic distribution of the test statistic. It returns
    # either 0 or 1, i.e. either rejects or accepts null hypothesis for given significance level θ.

    # Inputs:
        # pms_1, pms_2 : two samples of probability measures, each of which is a vector of Normal distributions
        # θ : significance level, default value is 0.05
        # n_boostrap: number of times we repeat bootstrap procedure to estimate the quantile of the test statistic


    # threshold is choosen from asymptotic distribution of test statistic which is chi-squared.

    n_top = length(pms_1)
    T_n = test_statistic_dm(pms_1, pms_2) # test statistic
    
    threshold = quantile(Chisq(1), 1 - θ) # obtain quantile from asymptotic distribution of test statistic
    decision = 1.0*(T_n > threshold) # 1.0 if T_n > threshold, 0.0 otherwise.
    return decision
end


function decide_dm_boostrap(pms_1::Vector{Float64}, pms_2::Vector{Float64}, n_boostrap::Int, θ = 0.05)
    # This function implements the two-sample test according to Dubey & Muller using boostrap approach. It returns
    # either 0 or 1, i.e. either rejects or accepts null hypothesis for given significance level θ.

    # Inputs:
        # pms_1, pms_2 : two samples of means of Normal distributions with variance 1
        # θ : significance level, default value is 0.05
        # n_boostrap: number of times we repeat bootstrap procedure to estimate the quantile of the test statistic

    # Threshold is obtained via boostrap procedure.
    n_top = length(pms_1)
    T_n = test_statistic_dm(pms_1, pms_2) # test statistic
    # obtain quantile using bootstrap approach
    T_n_boostrap = zeros(n_boostrap)
    for s in 1:n_boostrap
        allmeasures = vcat(pms_1, pms_2) # collect all probability measures into one vector
        pms_1_boostrap = sample(allmeasures, n_top; replace=true) # resample from pooled probability measures
        pms_2_boostrap = sample(allmeasures, n_top; replace=true) # resample from pooled probability measures
        T_n_boostrap[s] = test_statistic_dm(pms_1_boostrap, pms_2_boostrap) # test statistic from boostraped sample
    end
    threshold = quantile(T_n_boostrap, 1-θ)
    decision = 1.0*(T_n > threshold) # 1.0 if T_n > threshold, 0.0 otherwise.
    return decision
end



function decide_denanova_from_r(mu_1::Vector{Float64}, mu_2::Vector{Float64};
                           sd::Float64 = 1.0, nq::Int = 100, qmin=0.01, qmax=0.99,
                           n_boostrap::Int = 1000, seed = 1234, θ::Float64 = 0.05)

    # This function implements the two-sample test according Dubey & Muller using the function to DenANOVA from R package "frechet".
    # It returns either 0 or 1, i.e. either rejects or accepts null hypothesis for given significance level θ.

    # Inputs:
        # mu_1, mu_2 : two samples of means of Normal distributions with variance 1.
        # sd : standard deviation of each Normal distribution, default value is 1.0, we do not change it.
        # nq : number of grid points to approximate quantile functions, default value is 51
        # qmin, qmax : min and max quantiles to approximate quantile functions, default values are 0.01 and 0.99
        # θ : significance level, default value is 0.05
        # n_boostrap: number of times we repeat bootstrap procedure to estimate the quantile of the test statistic
        # seed : random seed for R, default value is 1234

    @rput mu_1 mu_2 sd nq qmin qmax n_boostrap seed

    R"""
    # if (!requireNamespace("frechet", quietly = TRUE)) {
    #   install.packages("frechet", repos="https://cloud.r-project.org")
    # }

    set.seed(seed)

    n1 <- length(mu_1)
    n2 <- length(mu_2)
    qSup <- seq(qmin, qmax, length.out = nq)

    Y1 <- lapply(seq_len(n1), function(i) qnorm(qSup, mean = mu_1[i], sd = sd))
    Y2 <- lapply(seq_len(n2), function(i) qnorm(qSup, mean = mu_2[i], sd = sd))

    Ly <- c(Y1, Y2)
    Lx <- qSup
    group <- c(rep(1, n1), rep(2, n2))

    res <- frechet::DenANOVA(qin = Ly, supin = Lx, group = group,
                    optns = list(boot = TRUE, R = n_boostrap))

    p_boot <- res$pvalBoot
    """
    @rget p_boot
    return 1 * (p_boot < θ)
end


# Energy statistic (Szekely & Rizzo 2004) 

function test_statistic_energy(pms_1::Vector{Normal}, pms_2::Vector{Normal})
    # This function computes the test statistic according to Energy statistic (Szekely & Rizzo 2004) for a given two samples of probobability measures
    # which are normal distributions.

    # Inputs:
        # pms_1, pms_2 : two samples of probability measures, each of which is a vector of Normal distributions

    # Recall that we assume that each of the sample is Normal distribution, and for W_2 metric we only need their means.
    n = length(pms_1)
    @assert n == length(pms_2) "two samples of probability measures must have same size "

    X = [pms_1[i].μ for i in 1:n] # collect all the means from first samples of probability measures
    Y = [pms_2[i].μ for i in 1:n] # collect all the means from second samples of probability measures

    distances_X = [abs(X[i] - X[j]) for i in 1:n, j in 1:n]
    distances_Y = [abs(Y[i] - Y[j]) for i in 1:n, j in 1:n]
    distances_XY = [abs(X[i] - Y[j]) for i in 1:n, j in 1:n]


    distance = 2 * mean(distances_XY) - mean(distances_X) - mean(distances_Y)
    T_n = distance * n / 2
    return T_n
end



function decide_energy_boostrap(pms_1::Vector{Normal}, pms_2::Vector{Normal}, n_boostrap::Int, θ = 0.05)
    # This function implements the two-sample test according to Energy statistic (Szekely & Rizzo 2004) using boostrap approach. It returns
    # either 0 or 1, i.e. either rejects or accepts null hypothesis for given significance level θ.

    # Inputs:
        # pms_1, pms_2 : two samples of probability measures, each of which is a vector of Normal distributions
        # θ : significance level, default value is 0.05
        # n_boostrap: number of times we repeat bootstrap procedure to estimate the quantile of the test statistic


    n_top = length(pms_1)
    T_n = test_statistic_energy(pms_1, pms_2) # test statistic
    # obtain quantile using bootstrap approach
    T_n_boostrap = zeros(n_boostrap)
    for s in 1:n_boostrap
        allmeasures = vcat(pms_1, pms_2) # collect all probability measures into one vector
        pms_1_boostrap = sample(allmeasures, n_top; replace=true) # resample from pooled probability measures
        pms_2_boostrap = sample(allmeasures, n_top; replace=true) # resample from pooled probability measures
        T_n_boostrap[s] = test_statistic_energy(pms_1_boostrap, pms_2_boostrap) # test statistic from boostraped sample
    end
    threshold = quantile(T_n_boostrap, 1-θ)
    decision = 1.0*(T_n > threshold) # 1.0 if T_n > threshold, 0.0 otherwise.
    return decision
end


# Methods for WoW and HIPM



function get_thresholds_permutation_hipm_wow(hier_sample_1::emp_ppm, hier_sample_2::emp_ppm, n_top::Int, n_bottom::Int, n_permutations::Int, θ::Float64)
    # This function gets the thresholds for HIPM and WoW distances using permutation approach. It receives hierarchical samples.

    # Inputs:
        # hier_sample_1, hier_sample_2 : hierarchical samples
        # n_top : number of atoms in hierarchical samples generated from RPMs
        # n_bottom : number of observations generated from each atom in hierarchical samples
        # n_permutations : number of permutations to approximate the quantile of the distance
        # θ : significance level, default value is 0.05
    
    permuted_samples_dlip = zeros(n_permutations) # store samples of distances
    permuted_samples_ww = zeros(n_permutations) # store samples of distances
    a = minimum([hier_sample_1.a, hier_sample_2.a])
    b = maximum([hier_sample_1.b, hier_sample_2.b])
    for k in 1:n_permutations
        total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
        random_indices = randperm(2*n_top) # indices to distribute rows to new hierarchical meausures

        atoms_1 = total_rows[random_indices[1:n_top],:] # first rows indexed by n_top random indices to the atoms_1
        atoms_2 = total_rows[random_indices[n_top+1:end],:] # first rows indexed by n_top random indices to the atoms_2

        hier_sample_1_permuted = emp_ppm(atoms_1, n_top, n_bottom, a, b)
        hier_sample_2_permuted = emp_ppm(atoms_2, n_top, n_bottom, a, b)

        
        permuted_samples_dlip[k] = dlip(hier_sample_1_permuted, hier_sample_2_permuted)
        permuted_samples_ww[k] = ww(hier_sample_1_permuted, hier_sample_2_permuted)
    end

    threshold_hipm = quantile(permuted_samples_dlip, 1 - θ)
    threshold_wow = quantile(permuted_samples_ww, 1 - θ)

    return threshold_hipm, threshold_wow
end



function get_thresholds_permutation_hipm_wow(q_1::PPM, q_2::PPM, n_top::Int, n_bottom::Int, n_permutations::Int, θ::Float64)
    # This function gets the thresholds for HIPM and WoW distances using permutation approach. It obtains hierarchical samples 
    # from two given RPMs and use them for permutation procedure.

    # Inputs:
        # q_1, q_2 : laws of two RPMs
        # n_top : number of atoms in hierarchical samples generated from RPMs
        # n_bottom : number of observations generated from each atom in hierarchical samples
        # n_permutations : number of permutations to approximate the quantile of the distance
        # θ : significance level, default value is 0.05

    hier_sample_1, hier_sample_2 = generate_emp(q_1, n_top, n_bottom), generate_emp(q_2, n_top, n_bottom) 
    return get_thresholds_permutation_hipm_wow(hier_sample_1, hier_sample_2, n_top, n_bottom, n_permutations, θ)
end
    





function get_thresholds_boostrap_hipm_wow(hier_sample_1::emp_ppm, hier_sample_2::emp_ppm, n_top::Int, n_bottom::Int, n_boostrap::Int, θ::Float64)
    # This function gets the thresholds for HIPM and WoW distances using boostrap approach. It receives hierarchical samples

    # Inputs:
        # hier_sample_1, hier_sample_2 : Hierarchical samples 
        # n_top : number of atoms in hierarchical samples generated from RPMs
        # n_bottom : number of observations generated from each atom in hierarchical samples
        # n_boostrap : number of boostrap samples to approximate the quantile of the distance
        # θ : significance level, default value is 0.05
    
 
    boostrap_samples_dlip = zeros(n_boostrap) # store samples of distances
    boostrap_samples_ww = zeros(n_boostrap) # store samples of distances
    a = minimum([hier_sample_1.a, hier_sample_2.a])
    b = maximum([hier_sample_1.b, hier_sample_2.b])
    for k in 1:n_boostrap
        total_rows = vcat(hier_sample_1.atoms, hier_sample_2.atoms) # collect all rows
        
        indices_1 = sample(1:2*n_top, n_top; replace = true)
        indices_2 = sample(1:2*n_top, n_top; replace = true)
        atoms_1 = total_rows[indices_1,:]  # resample from pooled hierarchical sample
        atoms_2 = total_rows[indices_2,:]  # resample from pooled hierarchical sample
        
    
        hier_sample_1_boostrap = emp_ppm(atoms_1, n_top, n_bottom, a, b)
        hier_sample_2_boostrap = emp_ppm(atoms_2, n_top, n_bottom, a, b)

        boostrap_samples_dlip[k] = dlip(hier_sample_1_boostrap, hier_sample_2_boostrap)
        boostrap_samples_ww[k] = ww(hier_sample_1_boostrap, hier_sample_2_boostrap)
    end

    threshold_hipm = quantile(boostrap_samples_dlip, 1 - θ)
    threshold_wow = quantile(boostrap_samples_ww, 1 - θ)

    return threshold_hipm, threshold_wow
end
    



function get_thresholds_boostrap_hipm_wow(q_1::PPM, q_2::PPM, n_top::Int, n_bottom::Int, n_boostrap::Int, θ::Float64)
    # This function gets the thresholds for HIPM and WoW distances using boostrap approach. It obtains hierarchical samples 
    # from two given RPMs and use them for boostrap procedure.

    # Inputs:
        # q_1, q_2 : laws of two RPMs
        # n_top : number of atoms in hierarchical samples generated from RPMs
        # n_bottom : number of observations generated from each atom in hierarchical samples
        # n_boostrap : number of boostrap samples to approximate the quantile of the distance
        # θ : significance level, default value is 0.05
    
    hier_sample_1, hier_sample_2 = generate_emp(q_1, n_top, n_bottom), generate_emp(q_2, n_top, n_bottom)
    return get_thresholds_boostrap_hipm_wow(hier_sample_1, hier_sample_2, n_top, n_bottom, n_boostrap, θ)
end
    



# Rejection rates

function rejection_rate(q_1::PPM, q_2::PPM, S::Int, n_top::Int, n_bottom::Int, n_boostrap::Int, θ::Float64, boostrap::Bool=true)
    # This function computes the rejection rates for given two laws of RPM, q_1 and q_2, for 4 testing schemes:
    # Dubey & Mueller, HIPM, WoW, Energy statistic

    # Input: 
        # q_1, q_2 : laws of two RPMs
        # S : number of times we simulate two samples from given laws of RPM
        # n_top : number of probability measures we simulate from each q
        # n_bottom : number of random variables we simulate from each of the probability measure from q
        # n_boostrap : number of times we repeat bootstrap procedure to estimate the quantile of the test statistic. Note that this can
        #              be number of permutations, depending whether boostrap variable is true or false but only for HIPM and WoW. For DM and
        #              Energy statistic, this is always number of boostrap samples.
        # θ : significance level, default value is 0.05
        # boostrap : if true, then thresholds for HIPM and WoW are obtained via boostrap approach, otherwise via permutation approach.

    # Output:
        # rej_rate_dm : rejection rate for Dubey & Mueller test
        # rej_rate_hipm : rejection rate for HIPM test
        # rej_rate_wow : rejection rate for WoW test
        # rej_rate_energy : rejection rate for Energy statistic test    


  
    rej_rate_dm, rej_rate_hipm, rej_rate_wow, rej_rate_energy = 0.0, 0.0, 0.0, 0.0
    #time_perm = time()

    # Instead of getting the threshold for HIPM and WoW based on hierarchical sample everytime, for fixed q_1 and q_2, we obtain
    # threshold once from some generated hierarchical samples and then use it for other hierarchical samples.
    if boostrap
        threshold_hipm, threshold_wow = get_thresholds_boostrap_hipm_wow(q_1, q_2, n_top, n_bottom, n_boostrap, θ) 
        #println("time taken to get thresholds is $(time() - time_perm) seconds")
    else
        threshold_hipm, threshold_wow = get_thresholds_permutation_hipm_wow(q_1, q_2, n_top, n_bottom, n_boostrap, θ) 
        #println("time taken to get thresholds is $(time() - time_perm) seconds")

    end
    #println("time taken to get thresholds is $(time() - time_perm) seconds")
    
    #time_S = time()
    for s in 1:S
        
        pms_1, pms_2 = generate_prob_measures(q_1, n_top), generate_prob_measures(q_2, n_top) # generate n_top probability measures
                                            # from q_1 and q_2

        mu_1, mu_2 = [pm_1.μ for pm_1 in pms_1], [pm_2.μ for pm_2 in pms_2] # collect means of all probability measures in pms_1 and pms_2

        hier_sample_1, hier_sample_2 = generate_emp(pms_1, n_top, n_bottom), generate_emp(pms_2, n_top, n_bottom) # generate n_bottom
                                            # random variables from each probability measures in pms_1 and pms_2
        # endpoints of the sample space for observatinos from hier_sample_1 and hier_sample_2 might be different, so we fix it
        a = minimum([hier_sample_1.a, hier_sample_2.a])
        b = maximum([hier_sample_1.b, hier_sample_2.b])
        
        hier_sample_1.a = a
        hier_sample_2.a = a

        hier_sample_1.b = b
        hier_sample_2.b = b

        # record if testing schemes reject
        rej_rate_dm += decide_denanova_from_r(mu_1, mu_2; n_boostrap=n_boostrap, θ = θ)  
        rej_rate_hipm += 1*(dlip(hier_sample_1, hier_sample_2) > threshold_hipm)
        rej_rate_wow += 1*(ww(hier_sample_1, hier_sample_2) > threshold_wow)

        rej_rate_energy += decide_energy_boostrap(pms_1, pms_2, n_boostrap, θ)
    end
    #println("time taken for S=$(S) is $(time() - time_S) seconds")
    rej_rate_dm /= S
    rej_rate_hipm /= S
    rej_rate_wow /= S
    rej_rate_energy /= S
    return rej_rate_dm,  rej_rate_hipm, rej_rate_wow, rej_rate_energy
end

# In the simulations, we consider the several pairs of laws of random probability measures for which we want to obtain the rejection rates.
# Such laws will be parametrized. Below we have a code that will record rejection rates for such pairs of laws of rpm per each parameter.

function rejection_rates_per_parameter(δs::Vector{Float64}, rej_rate_function::Function)
    # This function returns rejection rates using rej_rate_function for each each δ in δs. The parameter δ
    # is used to define two laws of RPMs, q_1 and q_2, for which we compute rejection rates.

    # Input:
        # δs : vector of Float64 numbers, each of which is used to define two laws of RPMs, q_1 and q_2
        # rej_rate_function : function which takes δ as input and returns rejection rates.
    
    rej_rates = zeros(length(δs), 4) # per each delta and testing scheme

    for i in 1:length(δs)
        println("parameter is $(δs[i])")
        rej_rates[i,:] .= rej_rate_function(δs[i]) # returns rejection rate for δ[i] for each testing scheme
    end
    return rej_rates
end




# Methods for saving computatino times 
function save_times(n_tops::Vector{Int}, n_bottoms::Vector{Int}, n_comps::Int)
    # This functions stores in a csv file the number of seconds we need to compute distances (HIPM, WoW) per each pair of (n_top, n_bottom) values.

    # Input:
        # n_tops: vector of number of probability measures
        # n_bottoms: vector of number of random variables from probability measures
        # n_comps: number of times we compute distance for each (n_top, n_bottom) pair. We will average the times over n_comps runs.
    
    
    
    # Firstly we define two Dirichlet Processes with different concentration parameters and the same base distribution.


    α_1, α_2 = 1.0, 2.0
    P_0_1 = ()->rand(Beta(1,1))
   

    a, b = -1.0, 1.0

    q_1 = DP(α_1, P_0_1, a, b)
    q_2 = DP(α_2, P_0_1, a, b)


    times_hipm = zeros(length(n_tops), length(n_bottoms)) # matrix to store average times for HIPM
    times_wow = zeros(length(n_tops), length(n_bottoms)) # matrix to store average times for WoW

    for (i, n_top) in enumerate(n_tops)
        for (j, n_bottom) in enumerate(n_bottoms)
            #println("n_top = $(n_top), n_bottom = $(n_bottom)")
            for s in 1:n_comps
                hier_sample_1 = generate_emp(q_1, n_top, n_bottom)
                hier_sample_2 = generate_emp(q_2, n_top, n_bottom)

                t_hipm = @elapsed dlip(hier_sample_1, hier_sample_2) # time HIPM
                t_wow = @elapsed ww(hier_sample_1, hier_sample_2) # time WoW

                times_hipm[i,j] += t_hipm
                times_wow[i,j] += t_wow
            end
            times_hipm[i,j] /= n_comps
            times_wow[i,j] /= n_comps
        end
    end
    # Round times to 1 decimal places for better readability
    # times_wow = Integer.(times_wow)
    # times_hipm = Integer.(times_hipm)
    times_wow = round.(times_wow, digits = 2)
    times_hipm = round.(times_hipm, digits = 2)


    # Build the DataFrame
    #df_hipm = DataFrame(times_hipm, Symbol.(string.("m = ", n_bottoms)))
    df_hipm = DataFrame(times_hipm, Symbol.(string.("m = ", n_bottoms)))
    df_hipm.n_tops = n_tops                  # add n_tops as a column
    rename!(df_hipm, :n_tops => :n)
    select!(df_hipm, :n, :)             # move n_tops to the first column

    #df_wow = DataFrame(times_wow, Symbol.(string.("m = ", n_bottoms)))
    df_wow = DataFrame(times_wow, Symbol.(string.("m = ", n_bottoms)))
    df_wow.n_tops = n_tops                  # add n_tops as a column
    rename!(df_wow, :n_tops => :n)
    select!(df_wow, :n, :)             # move n_tops to the first column

    filepath = joinpath(pwd(), "time_wow_hipm/")
    CSV.write(filepath*"times_hipm_powers_of_two.csv", df_hipm)
    CSV.write(filepath*"times_wow_powers_of_two.csv", df_wow)    


    return times_hipm, times_wow
end

# Methods for comparing powers for each pair 





function rejection_rate_hipm_wow(q_1::PPM, q_2::PPM, S::Int, n_top::Int, n_bottom::Int, n_boostrap::Int, θ::Float64, boostrap::Bool=true)
    # This function computes the rejection rates for given two laws of RPM, q_1 and q_2, for 2 testing schemes:
    # HIPM, WoW

    # Input: 
        # q_1, q_2 : laws of two RPMs
        # S : number of times we simulate two samples from given laws of RPM
        # n_top : number of probability measures we simulate from each q
        # n_bottom : number of random variables we simulate from each of the probability measure from q
        # n_boostrap : number of times we repeat bootstrap procedure to estimate the quantile of the test statistic. Note that this can
        #              be number of permutations, depending whether boostrap variable is true or false but only for HIPM and WoW. 
        # θ : significance level, default value is 0.05
        # boostrap : if true, then thresholds for HIPM and WoW are obtained via boostrap approach, otherwise via permutation approach.

    # Output:
        # rej_rate_hipm : rejection rate for HIPM test
        # rej_rate_wow : rejection rate for WoW test
     

  
    rej_rate_hipm, rej_rate_wow = 0.0, 0.0
    #time_perm = time()

    # Instead of getting the threshold for HIPM and WoW based on hierarchical sample everytime, for fixed q_1 and q_2, we obtain
    # threshold once from some generated hierarchical samples and then use it for other hierarchical samples.
    if boostrap
        threshold_hipm, threshold_wow = get_thresholds_boostrap_hipm_wow(q_1, q_2, n_top, n_bottom, n_boostrap, θ) 
    else
        threshold_hipm, threshold_wow = get_thresholds_permutation_hipm_wow(q_1, q_2, n_top, n_bottom, n_boostrap, θ) 
    end
    #println("time taken to get thresholds is $(time() - time_perm) seconds")
    
    #time_S = time()
    for s in 1:S

        hier_sample_1, hier_sample_2 = generate_emp(q_1, n_top, n_bottom), generate_emp(q_2, n_top, n_bottom) 
        if s % 5 == 0
            if boostrap
                threshold_hipm, threshold_wow = get_thresholds_boostrap_hipm_wow(hier_sample_1, hier_sample_2, n_top, n_bottom, n_boostrap, θ) 
            else
                threshold_hipm, threshold_wow = get_thresholds_permutation_hipm_wow(hier_sample_1, hier_sample_2, n_top, n_bottom, n_boostrap, θ) 
            end
        end

        # record if testing schemes reject
        rej_rate_hipm += 1*(dlip(hier_sample_1, hier_sample_2) > threshold_hipm)
        rej_rate_wow += 1*(ww(hier_sample_1, hier_sample_2) > threshold_wow)
    end
    #println("time taken for S=$(S) is $(time() - time_S) seconds")
    rej_rate_hipm /= S
    rej_rate_wow /= S
    return rej_rate_hipm, rej_rate_wow
end


function savefig_for_rejections_hipm_wow(q_1::PPM, q_2::PPM, n_tops::Vector{Int}, n_bottoms::Vector{Int}, S::Int, n_permutations::Int, θ::Float64, boostrap::Bool=true, name::String="dirichlet")
    # This function computes rejection rates per each pair (n, m) for laws of RPM q_1 and q_2. It stores rejection rates in csv filemode

    # Inputs:
        # q_1
    
    rejections_hipm, rejections_wow = zeros(length(n_tops), length(n_bottoms)), zeros(length(n_tops), length(n_bottoms))

    for (i,n_top) in enumerate(n_tops)
        for (j,n_bottom) in enumerate(n_bottoms)
            println("n_top = $(n_top), n_bottom = $(n_bottom)")
            rej_rate_hipm, rej_rate_wow = rejection_rate_hipm_wow(q_1, q_2, S, n_top, n_bottom, n_permutations, θ, boostrap)
            rejections_hipm[i,j] = rej_rate_hipm
            rejections_wow[i,j] = rej_rate_wow
        end
    end

    rejections_wow = round.(rejections_wow, digits = 3)
    rejections_hipm = round.(rejections_hipm, digits = 3)


    # Build the DataFrame
    df_rejections_hipm = DataFrame(rejections_hipm, Symbol.(string.("m = ", n_bottoms)))
    df_rejections_hipm.n_tops = n_tops                  # add n_tops as a column
    rename!(df_rejections_hipm, :n_tops => :n)
    select!(df_rejections_hipm, :n, :)             # move n_tops to the first column

    df_rejections_wow = DataFrame(rejections_wow, Symbol.(string.("m = ", n_bottoms)))
    df_rejections_wow.n_tops = n_tops                  # add n_tops as a column
    rename!(df_rejections_wow, :n_tops => :n)
    select!(df_rejections_wow, :n, :)             # move n_tops to the first column


    filepath = joinpath(pwd(), "rejections_n_vs_m_wow_hipm/")
    CSV.write(filepath*"wow/new/df_rejections_wow_$(name).csv", df_rejections_wow)
    CSV.write(filepath*"hipm/new/df_rejections_hipm_$(name).csv", df_rejections_hipm)





    directions_wow = -1 * ones(length(n_tops), length(n_bottoms))
    directions_hipm = -1 * ones(length(n_tops), length(n_bottoms))
    for i in 1:(length(n_tops) - 1)
        for j in 1:(length(n_bottoms) - 1)
            if rejections_wow[i, j + 1] > rejections_wow[i + 1, j]
                directions_wow[i, j] = 1 # it is better to increase m/n_bottom 2 times.
            elseif rejections_wow[i, j + 1] < rejections_wow[i + 1, j]
                directions_wow[i, j] = 0
            else
                directions_wow[i, j] = 5
            end
            if rejections_hipm[i, j + 1] > rejections_hipm[i + 1, j]
                directions_hipm[i, j] = 1 # it is better to increase m/n_bottom 2 times.
            elseif rejections_hipm[i, j + 1] < rejections_hipm[i + 1, j]
                directions_hipm[i, j] = 0
            else
                directions_hipm[i, j] = 5
            end
        end
    end

    # Build the DataFrame
    df_directions_hipm = DataFrame(directions_hipm, Symbol.(string.("m = ", n_bottoms)))
    df_directions_hipm.n_tops = n_tops                  # add n_tops as a column
    rename!(df_directions_hipm, :n_tops => :n)
    select!(df_directions_hipm, :n, :)             # move n_tops to the first column

    df_directions_wow = DataFrame(directions_wow, Symbol.(string.("m = ", n_bottoms)))
    df_directions_wow.n_tops = n_tops                  # add n_tops as a column
    rename!(df_directions_wow, :n_tops => :n)
    select!(df_directions_wow, :n, :)             # move n_tops to the first column


    filepath = joinpath(pwd(), "rejections_n_vs_m_wow_hipm/")
    CSV.write(filepath*"wow/new/df_directions_wow_$(name).csv", df_directions_wow)
    CSV.write(filepath*"hipm/new/df_directions_hipm_$(name).csv", df_directions_hipm)


    return rejections_wow, rejections_hipm
end


