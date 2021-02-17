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
    covariates,        # [Vector{<:AbstractMatrix}]
    covariates_partial_regional,
    covariates_partial_state,
    epidemic_start,    # [AbstractVector{<:Int}]
    population,        # [AbstractVector{<:Real}]
    serial_intervals,  # [AbstractVector{<:Real}] fixed pre-calculated serial interval (SI) using empirical data from Neil
    region,
    num_weeks,
    week_index,
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
    ϕ ~ truncated(Normal(0, 5), 0, Inf)
    κ ~ truncated(Normal(0, 0.5), 0, Inf)
    μ ~ filldist(truncated(Normal(3.28, κ), 0, Inf), num_states)
    α ~ filldist(Normal(0,0.5), num_covariates)
    γ_region ~ truncated(Normal(0,0.5), 0, Inf)
    α_region ~ filldist(Normal(0,γ_region), num_regions, num_covariates_regional)
    γ_state  ~ truncated(Normal(0,0.5), 0, Inf)
    α_state  ~ filldist(Normal(0,γ_state), num_states, num_covariates_state)
    ifr_noise ~ filldist(truncated(Normal(1., 0.1), 0, Inf), num_states)

    ρ1 ~ truncated(Normal(0.8,0.05), 0, 1)
    ρ2 ~ truncated(Normal(0.1,0.05), 0, 1)
    σw ~ truncated(Normal(0,0.2), 0, Inf)
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
    expected_daily_cases = TV[TV(undef, last_time_steps[m]) for m in 1:num_states]
    cases_pred = TV[TV(undef, last_time_steps[m]) for m in 1:num_states]
    expected_daily_deaths = TV[TV(undef, last_time_steps[m]) for m in 1:num_states]
    Rt = TV[TV(undef, last_time_steps[m]) for m in 1:num_states]
    Rt_adj = TV[TV(undef, last_time_steps[m]) for m in 1:num_states]
    infectiousness = TV[TV(undef, last_time_steps[m]) for m in 1:num_states]
    # Loops over states and perform independent computations for each country
    # since this model does not include any notion of migration across borders.
    # => might has well wrap it in a `@threads` to perform the computation in parallel.
    @threads for m = 1:num_states
        # Country-specific parameters
        ifr_noise_m = ifr_noise[m]
        π_m = π[m]
        pop_m = population[m]
        expected_daily_cases_m = expected_daily_cases[m]
        cases_pred_m = cases_pred[m]
        expected_daily_deaths_m = expected_daily_deaths[m]
        Rt_m = Rt[m]
        Rt_adj_m = Rt_adj[m]
        weekly_effect_m = weekly_effect[m]
        last_time_step = last_time_steps[m]
        infectiousness_m = infectiousness[m]
        week_index_m = week_index[m][1:last_time_step]

        # Imputation of `num_impute` days
        expected_daily_cases_m[1:num_impute] .= y[m]
        cases_pred_m[1] = zero(cases_pred_m[1])
        cases_pred_m[2:num_impute] .= cumsum(expected_daily_cases_m[1:num_impute - 1])

        xs = covariates[m][1:last_time_step, :] # extract covariates for the wanted time-steps and country `m`
        ys = covariates_partial_regional[m][1:last_time_step, :] # extract covariates for the wanted time-steps and country `m`
        zs = covariates_partial_state[m][1:last_time_step, :] # extract covariates for the wanted time-steps and country `m`
        Rt_m .= μ[m] * 2 * logistic.(
                    xs * (-α)
                  - ys * α_region[region[m],:]
                  - zs * α_state[m,:]
                  - weekly_effect_m[week_index_m]);

        # Adjusts for portion of pop that are susceptible
        # no adjustment in original paper: Rt_adj_m[1:num_impute] .= Rt_m[1:num_impute]
        Rt_adj_m[1:num_impute] .= (max.(pop_m .- cases_pred_m[1:num_impute], zero(cases_pred_m[1])) ./ pop_m) .* Rt_m[1:num_impute]
        infectiousness_m[1] = zero(expected_daily_cases_m[1])
        for t = 2:num_impute
            infectiousness_m[t] = sum(expected_daily_cases_m[τ] * serial_intervals[t - τ] for τ = 1:(t - 1))
        end
        for t = (num_impute + 1):last_time_step
            # Update cumulative cases
            cases_pred_m[t] = cases_pred_m[t - 1] + expected_daily_cases_m[t - 1]

            # Adjusts for portion of pop that are susceptible
            Rt_adj_m[t] = (max(pop_m - cases_pred_m[t], zero(cases_pred_m[t])) / pop_m) * Rt_m[t]

            convolution = sum(expected_daily_cases_m[τ] * serial_intervals[t - τ] for τ = 1:(t - 1))
            expected_daily_cases_m[t] = Rt_adj_m[t] * convolution
            infectiousness_m[t] = convolution
        end
        infectiousness_m ./= max_SI

        expected_daily_deaths_m[1] = 1e-15 * expected_daily_cases_m[1]
        for t = 2:last_time_step
            expected_daily_deaths_m[t] = ifr_noise_m * sum(expected_daily_cases_m[τ] * π_m[t - τ] for τ = 1:(t - 1))
        end
    end

    # Observe
    # Doing observations in parallel provides a small speedup
    @threads for m = 1:num_states
        # Extract the estimated expected daily deaths for country `m`
        expected_daily_deaths_m = expected_daily_deaths[m]
        # Extract time-steps for which we have observations
        ts = epidemic_start[m]:num_obs_states[m]
        # Observe!
        deaths[m][ts] ~ arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ))
    end
    return (
        expected_daily_cases = expected_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        Rt = Rt,
        Rt_adjusted = Rt_adj,
        infectiousness = infectiousness
    )
end
