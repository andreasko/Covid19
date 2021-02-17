# Replication of https://github.com/ImperialCollegeLondon/covid19model
using Turing
using Distributions, StatsBase
using StatsFuns
using ArgCheck
using FillArrays

using Base.Threads

import Turing: filldist

# This is the most up-to-date one
@model function model_v1(
    num_impute,        # [Int] num. of days for which to impute infections
    num_total_days,    # [Int] days of observed data + num. of days to forecast
    cases,             # [AbstractVector{<:AbstractVector{<:Int}}] reported cases
    deaths,            # [AbstractVector{<:AbstractVector{<:Int}}] reported deaths; rows indexed by i > N contain -1 and should be ignored
    π,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    π2,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    covariates,        # [Vector{<:AbstractMatrix}]
    covariates_partial_regional, # [Vector{<:AbstractMatrix}] covariates for partial pooling (region-level effects)
    covariates_partial_state, # [Vector{<:AbstractMatrix}] covariates for partial pooling (state-level effects)
    epidemic_start,    # [AbstractVector{<:Int}]
    population,        # [AbstractVector{<:Real}]
    serial_intervals,  # [AbstractVector{<:Real}] fixed pre-calculated serial interval (SI) using empirical data from Neil
    region,            # [Vector{Int64}] Macro region index for each state
    num_weeks,         # [Int] number of weeks (21)
    week_index,        # [Vector{Array{Int64,1}}] Macro region index for each state
    cases_start,       # [Int] 21 = max_date ("2020-05-11") - testing_date (2020-05-11) see publication
    predict=false,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    ::Type{TV} = Vector{Float64}
) where {TV}
    # `covariates` should be of length `num_states` and each entry correspond to a matrix of size `(num_total_days, num_covariates)`
    num_covariates_regional = size(covariates_partial_regional[1], 2)
    num_covariates_state = size(covariates_partial_state[1], 2)
    num_covariates = size(covariates[1], 2)
    num_states = length(cases)
    num_obs_states = length.(cases)
    num_regions = region |> unique |> length
    max_SI = maximum(serial_intervals)
    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_states[m]`
    last_time_steps = predict ? fill(num_total_days, num_states) : num_obs_states

    # Latent variables
    τ ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
    y = fill(TV(undef, num_impute), num_states)
    for m = 1:num_states
        y[m] .~ Exponential(τ)
    end
    ϕ         ~ truncated(Normal(0, 5), 0, Inf)
    ϕ2        ~ truncated(Normal(0, 5), 0, Inf)
    κ         ~ truncated(Normal(0, 0.5), 0, Inf)
    μ         ~ filldist(truncated(Normal(3.28, κ), 0, Inf), num_states)
    α         ~ filldist(Normal(0,0.5), num_covariates)
    γ_region  ~ truncated(Normal(0,0.5), 0, Inf)
    α_region  ~ filldist(Normal(0,γ_region), num_regions, num_covariates_regional)
    γ_state   ~ truncated(Normal(0,0.5), 0, Inf)
    α_state   ~ filldist(Normal(0,γ_state), num_states, num_covariates_state)
    ifr_noise ~ filldist(truncated(Normal(1., 0.1), 0, Inf), num_states)
    ρ1        ~ truncated(Normal(0.8,0.05), 0, 1)
    ρ2        ~ truncated(Normal(0.1,0.05), 0, 1)
    σw        ~ truncated(Normal(0,0.2), 0, Inf)
    under_reporting ~ Beta(12,5)

    #WARNING √x can fail during warm up as we dont guarantee x>0
    σw_star = σw * sqrt( (1+ρ2)*(1-ρ1-ρ2)*(1+ρ1-ρ2)/(1-ρ2) ) #sqrt(1-ρ1^2-ρ2^2-2ρ1*ρ2/(1-ρ2))
    weekly_effect = TV[TV(undef, num_weeks+1) for m in 1:num_states]
    for m in 1:num_states #TODO parallize?
        weekly_effect_m = weekly_effect[m]
        weekly_effect_m[1] ~ Normal(0,0.01)
        weekly_effect_m[2] ~ Normal(0,σw_star)
        for w in 3:num_weeks+1
            weekly_effect_m[w] ~ Normal(ρ1*weekly_effect_m[w-1]+ρ2*weekly_effect_m[w-2], σw_star)
        end
    end

    # Initialization of some quantities
    predicted       = TV[TV(undef, last_time_steps[m]) for m in 1:num_states]
    cumcases        = TV[TV(undef, last_time_steps[m]) for m in 1:num_states]
    expected_deaths = TV[TV(undef, last_time_steps[m]) for m in 1:num_states]
    expected_cases  = TV[TV(undef, last_time_steps[m]) for m in 1:num_states]
    infectiousness  = TV[TV(undef, last_time_steps[m]) for m in 1:num_states]
    Rt              = TV[TV(undef, last_time_steps[m]) for m in 1:num_states]
    Rt_adj          = TV[TV(undef, last_time_steps[m]) for m in 1:num_states]
    # Loops over states and perform independent computations for each country
    # since this model does not include any notion of migration across borders.
    # => might has well wrap it in a `@threads` to perform the computation in parallel.
    @threads for m = 1:num_states
        # Country-specific parameters
        ifr_noise_m       = ifr_noise[m]
        π_m               = π[m]
        π2_m              = π2[m]
        pop_m             = population[m]
        predicted_m       = predicted[m]
        cumcases_m        = cumcases[m]
        expected_deaths_m = expected_deaths[m]
        expected_cases_m  = expected_cases[m]
        Rt_m              = Rt[m]
        Rt_adj_m          = Rt_adj[m]
        weekly_effect_m   = weekly_effect[m]
        last_time_step    = last_time_steps[m]
        infectiousness_m  = infectiousness[m]
        week_index_m      = week_index[m][1:last_time_step]

        # Imputation of `num_impute` days
        predicted_m[1:num_impute] .= y[m]
        cumcases_m[1] = zero(cumcases_m[1])
        cumcases_m[2:num_impute] .= cumsum(predicted_m[1:num_impute - 1])

        xs    = covariates[m][1:last_time_step, :] # extract covariates for the wanted time-steps and country `m`
        ys    = covariates_partial_regional[m][1:last_time_step, :] # extract covariates for the wanted time-steps and country `m`
        zs    = covariates_partial_state[m][1:last_time_step, :] # extract covariates for the wanted time-steps and country `m`
        Rt_m .= μ[m] * 2 * logistic.(
                    xs * (-α)
                  - ys * α_region[region[m],:]
                  - zs * α_state[m,:]
                  - weekly_effect_m[week_index_m]);

        # Adjusts for portion of pop that are susceptible
        # no adjustment in original paper: Rt_adj_m[1:num_impute] .= Rt_m[1:num_impute]
        Rt_adj_m[1:num_impute] .= (max.(pop_m .- cumcases_m[1:num_impute], zero(cumcases_m[1])) ./ pop_m) .* Rt_m[1:num_impute]
        infectiousness_m[1] = zero(predicted_m[1])
        for t = 2:num_impute
            infectiousness_m[t] = sum(predicted_m[τ] * serial_intervals[t - τ] for τ = 1:(t - 1))
        end
        for t = (num_impute + 1):last_time_step
            # Update cumulative cases
            cumcases_m[t] = cumcases_m[t - 1] + predicted_m[t - 1]

            # Adjusts for portion of pop that are susceptible
            Rt_adj_m[t] = (max(pop_m - cumcases_m[t], zero(cumcases_m[t])) / pop_m) * Rt_m[t]

            convolution = sum(predicted_m[τ] * serial_intervals[t - τ] for τ = 1:(t - 1))
            predicted_m[t] = Rt_adj_m[t] * convolution
            infectiousness_m[t] = convolution
        end
        infectiousness_m ./= max_SI

        expected_deaths_m[1] = 1e-15 * predicted_m[1]
        for t = 2:last_time_step
            expected_deaths_m[t] = ifr_noise_m * sum(predicted_m[τ] * π_m[t - τ] for τ = 1:(t - 1))
            expected_cases_m[t]  = sum(predicted_m[τ] * π2_m[t - τ] for τ = 1:(t - 1))
        end
    end

    # Observe
    # Doing observations in parallel provides a small speedup
    for m = 1:num_states
        # Extract the estimated expected_cases daily deaths for country `m`
        expected_deaths_m = expected_deaths[m]
        num_obs_states_m  = num_obs_states[m]
        # Extract time-steps for which we have observations
        ts = epidemic_start[m]:num_obs_states_m
        # Observe!
        deaths[m][ts] ~ arraydist(NegativeBinomial2.(expected_deaths_m[ts], ϕ))
        if cases_start > 0
            ts = (num_obs_states_m-cases_start):num_obs_states_m
            cases[m][ts] ~ arraydist(NegativeBinomial2.(expected_deaths_m[ts] * under_reporting[m], ϕ2))
        end
    end
    return (
        expected_daily_cases = expected_cases,
        expected_daily_deaths = expected_deaths,
        predicted_daily_cases = predicted,
        Rt = Rt,
        Rt_adjusted = Rt_adj,
        infectiousness = infectiousness
    )
end
