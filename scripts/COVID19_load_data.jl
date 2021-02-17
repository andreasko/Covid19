using DrWatson
quickactivate(@__DIR__)
@show projectdir()
using CSV
using DataFrames
using DataFramesMeta
using Dates
using Covid19
#----------------------------------------------------------------------------
# load data from Imperial College
data = ImperialReport13.load_data(datadir("imperial-report13", "processed.rds"))
turing_data = data.turing_data



#---------------------------------------------------------------------------
dk = CSV.read("/home/and/code/Covid19/data/COVID19/DK.csv")
# dk = @where(dk_full, in(dk_dates).(:date))
tryparsem(T, str) = something(tryparse(T, str), missing)

function fill_with_previous!(xs)
    N = length(xs)
    xs[1] == "NA" && (xs[1] = "0")
    for i in 2:N
        xs[i] == "NA" && (xs[i] = xs[i-1])
    end
end

function parse_cumulative(c)
    d = replace(c, "NA"=>"0")
    d = tryparse.(Int, d)
    Δd = diff(d)
    Δd[Δd .< 0] .= 0
    Δd[1:end]
end

covid19_data, dates, covariate_names = let
    offset = Day(1) #offset between COVID19 and Imperial data
    m = 1
    Npred = 14
    first_i = findfirst(==(data.country_to_dates["Denmark"][1] - offset), dk.date)

    cases = [parse_cumulative(dk.confirmed[first_i-1:end])]
    Nobs = length(cases[1])
    Ntot = num_total_days = Nobs + Npred
    dates = dk.date[first_i:end] .+ offset
    deaths = [parse_cumulative(dk.deaths[first_i-1:end])]
    num_impute = 6
    π = zeros(Ntot)
    π[1:length(turing_data.π[m])] = turing_data.π[m]
    π = [π]
    population = [dk.population[1]]
    num_countries = 1
    num_obs_countries = [Nobs]
    epidemic_start = [31]
    serial_intervals = zeros(Ntot)
    serial_intervals[1:length(turing_data.serial_intervals)] = turing_data.serial_intervals

    covariates = hcat(
        dk.school_closing,
        dk.transport_closing,
        dk.workplace_closing,
        dk.cancel_events,
        dk.stay_home_restrictions,
        dk.gatherings_restrictions,
        dk.internal_movement_restrictions,
        # dk.international_movement_restrictions
    )
    for xs in eachcol(covariates) #fill latest unknown values assuming policies didnt change
        fill_with_previous!(xs)
    end
    covariates = tryparse.(Int, covariates)
    covariates = hcat(covariates, sum(covariates, dims=2) .> 0)
    covariates = vcat(covariates,
        repeat(covariates[end:end, 1:end], Npred))
    covariates = [covariates[first_i:end,:]]
    covariate_names = ["school_closing", "transport_closing", "workplace_closing", "cancel_events", "stay_home_restrictions", "gatherings_restrictions", "internal_movement_restrictions", "any"]
    (; num_countries, num_impute, num_obs_countries, num_total_days, cases, deaths, π, epidemic_start, population, serial_intervals, covariates), dates, Tuple(covariate_names)
end

struct Data
    stan_data
    turing_data
    country_to_dates
    reported_cases
    countries
    covariate_names
end

dkdata = ImperialReport13.Data(
    Dict(),
    covid19_data,
    Dict("Denmark" => dates),
    Dict("Denmark" => covid19_data.cases[1]),
    ("Denmark",),
    covariate_names
)

#-------------------------------------------------------------------------
using Plots
let
    plot(data.country_to_dates["Denmark"], cumsum(turing_data.cases[1]); label="Imperial")
    plot!(data.country_to_dates["Denmark"], cumsum(data.reported_cases["Denmark"]); label="reported cases")
    plot!(dates[1:100], cumsum(covid19_data.cases[1][1:100]), label="COVID19")
end

let
    plot(data.country_to_dates["Denmark"][1:64], turing_data.covariates[1][1:64,5]; label="Imperial")
    plot!(dates[1:100], covid19_data.covariates[1][1:100,3], label="COVID19")
end
#-------------------------------------------------------------------------
model_def = ImperialReport13.model_v2
lockdown_index = findfirst(==("workplace_closing"), covariate_names)
m = model_def(
    covid19_data.num_impute,
    covid19_data.num_total_days,
    covid19_data.cases,
    covid19_data.deaths,
    covid19_data.π,
    covid19_data.covariates,
    covid19_data.epidemic_start,
    covid19_data.population,
    covid19_data.serial_intervals,
    lockdown_index,
    false # <= DON'T predict
);
#---------------------------------------------------------------------------
#prior check
using Turing
#using Turing: Variational

res = m()
res.expected_daily_cases[1]
chain_prior = sample(m, Turing.Inference.Prior(), 4_000);
## plot results
using Plots, StatsPlots, LaTeXStrings
pyplot()
plot(chain_prior[[:ϕ, :τ, :κ]]; α = .5, linewidth=1.5)

generated_prior = vectup2tupvec(generated_quantities(m, chain_prior));
daily_cases_prior, daily_deaths_prior, Rt_prior, Rt_adj_prior = generated_prior; # <= tuple of `Vector{<:Vector{<:Vector}}`
ImperialReport13.country_prediction_plot(dkdata, 1, daily_cases_prior, daily_deaths_prior, Rt_prior; main_title = "(prior)")

#---------------------------------------------------------------------------
# posterior sampling
parameters = (
    warmup = 1000,
    steps =  1000
);

chains_posterior = sample(m, NUTS(parameters.warmup, 0.95; max_depth=10), parameters.steps + parameters.warmup)
## plot
plot(chains_posterior[[:ϕ, :τ, :κ]]; α = .5, linewidth=1.5)
generated_posterior = vectup2tupvec(generated_quantities(m, chains_posterior));
daily_cases, daily_deaths, Rt, Rt_adj = generated_posterior; # <= tuple of `Vector{<:Vector{<:Vector}}`
p =ImperialReport13.country_prediction_plot(dkdata, 1, daily_cases, daily_deaths, Rt; main_title = "(posterior)")
savefig(p, projectdir("figures/")*"dk_test.png")
##


chains_posterior_vec = [read(fname, Chains) for fname in filenames]; # Read the different chains
chains_posterior = chainscat(chains_posterior_vec...); # Concatenate them
chains_posterior = chains_posterior[1:3:end]
