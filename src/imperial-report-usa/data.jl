using RData, ArgCheck, DrWatson

using ..Covid19: rename!

struct Data
    stan_data
    turing_data
    state_to_dates
    reported_cases
    states
end

#-----------------------------------------------------------------------------
function _fix_stan_data_types!(d)
    # Convert some misparsed fields
    d["N2"] = Int(d["N2"])
    d["N0"] = Int(d["N0"])

    d["EpidemicStart"] = Int.(d["EpidemicStart"])

    # Stan will fail if these are `nothing` so we make them empty arrays
    d["x"] = []
    d["features"] = []

    # Add some type-information to arrays (as `-1` is supposed to represent, well, missing data)
    d["deaths"] = Int.(d["deaths"])
    d["cases"] = Int.(d["cases"])

    d["P"] = Int(d["P"]) # `num_covariates`
    d["P_partial_regional"] = Int(d["P_partial_regional"]) # `num_covariates`
    d["P_partial_state"] = Int(d["P_partial_state"]) # `num_covariates`
    d["M"] = Int(d["M"]) # `num_states`
    d["N0"] = Int(d["N0"]) # `num_impute`
    d["N"] = Int.(d["N"]) # `num_obs_states`
    d["N2"] = Int(d["N2"]) # `num_total_days`

    d["pop"] = Int.(d["pop"]) # `population`
    d["W"] = Int(d["W"])
    d["Q"] = Int(d["Q"])
    d["Region"] = Int.(d["Region"])
    d["week_index"] = Int.(d["week_index"])

    return d
end




#-----------------------------------------------------------------------------
defaultpath = "/home/and/code/ImperialCollegeReplications/FLA20a/Covid19/data/imperial-report-usa/processed.rds"
function load_data(path = defaultpath)
    @argcheck endswith(path, ".rds") "$(path) is not a RDS file"
    rdata_full = load(path)

    country_to_dates = Dict([(k, rdata_full["dates"][k]) for k in keys(rdata_full["dates"])])
    reported_cases = Dict([(k, Int.(rdata_full["reported_cases"][k])) for k in keys(rdata_full["reported_cases"])])
    reported_deaths = Dict([(k, Int.(rdata_full["reported_deaths"][k])) for k in keys(rdata_full["reported_cases"])])
    rdata = rdata_full["stan_data"]

    states = rdata_full["states"]
    num_states = length(states)

    # `rdata` is a `DictOfVector` so we convert to a simple `Dict` for simplicity
    # NOTE: `values(df)` and `keys(df)` have different ordering so DON'T do `Dict(keys(df), values(df))`
    d = Dict([(k, rdata[k]) for k in keys(rdata)])
    _fix_stan_data_types!(d)

    stan_data = copy(d)

    # Rename some columns
    rename!(
        d,
        "f" => "π", "SI" => "serial_intervals", "pop" => "population",
        "M" => "num_states", "N0" => "num_impute", "N" => "num_obs_states",
        "N2" => "num_total_days", "EpidemicStart" => "epidemic_start",
        "X" => "covariates", "X_partial_state" => "covariates_partial_state",
        "X_partial_regional" => "covariates_partial_regional", "P" => "num_covariates",
        "W" => "num_weeks"
    )
    d["deaths"] = collect(eachcol(d["deaths"])) # convert into Array of arrays instead of matrix
    d["cases"] = collect(eachcol(d["cases"])) # convert into Array of arrays instead of matrix
    d["π"] = collect(eachcol(d["π"]))  # convert into Array of arrays instead of matrix

    # Can deal with ragged arrays, so we can shave off unobserved data (future) which are just filled with -1
    num_obs_states = d["num_obs_states"]
    d["cases"] = collect(d["cases"][m][1:num_obs_states[m]] for m = 1:num_states)
    d["deaths"] = collect(d["deaths"][m][1:num_obs_states[m]] for m = 1:num_states)
    d["week_index"] = [d["week_index"][m, :] for m = 1:num_states]

    # Convert 3D array into Array{Matrix}
    covariates = [rdata["X"][m, :, :] for m = 1:num_states]
    covariates_partial_state = [rdata["X_partial_state"][m, :, :] for m = 1:num_states]
    covariates_partial_regional = [rdata["X_partial_regional"][m, :, :] for m = 1:num_states]

    turing_data = (; (k => d[String(k)] for k in [:num_states, :num_impute, :num_obs_states,
                                           :num_total_days, :cases, :deaths, :π, :epidemic_start,
                                           :population, :serial_intervals, :week_index, :Region, :num_weeks])...)
    turing_data = merge(turing_data, (; covariates, covariates_partial_state, covariates_partial_regional));

    return Data(stan_data, turing_data, country_to_dates, reported_cases, states)
end
