using DrWatson
@quickactivate
using ArgParse

argtable = ArgParseSettings(
    description="This script samples from ImperialReportUSA.model using NUTS."
)
@add_arg_table! argtable begin
    "--chunksize"
        help = "chunksize to be used by ForwardDiff.jl"
        arg_type = Int
        default = 40
    "--num-samples", "-n"
        help = "number of samples"
        arg_type = Int
        default = 3000
    "--num-warmup", "-w"
        help = "number of samples to use for warmup/adaptation"
        arg_type = Int
        default = 1000
    "seed"
        help = "random seed to use"
        required = true
        arg_type = Int
end

parsed_args = parse_args(ARGS, argtable)

# Loading the project (https://github.com/TuringLang/Covid19)
using Covid19

# Some other packages we'll need
using Random, Dates, Turing, Bijectors

using Base.Threads
nthreads()
#----------------------------------------------------------------------------
data = ImperialUSAcases.load_data(datadir("imperial-usa-cases", "processed.rds"))
turing_data = data.turing_data;
#----------------------------------------------------------------------------
function select_state(turing_data, idx)
    (
        num_states                  = 1,
        num_impute                  = turing_data.num_impute,
        num_obs_states              = [turing_data.num_obs_states[idx]],
        num_total_days              = turing_data.num_total_days,
        cases                       = [turing_data.cases[idx]],
        deaths                      = [turing_data.deaths[idx]],
        π                           = [turing_data.π[idx]],
        π2                          = [turing_data.π2[idx]],
        epidemic_start              = [turing_data.epidemic_start[idx]],
        population                  = [turing_data.population[idx]],
        serial_intervals            = turing_data.serial_intervals,
        week_index                  = [turing_data.week_index[idx]],
        Region                      = turing_data.Region,
        num_weeks                   = turing_data.num_weeks,
        covariates                  = [turing_data.covariates[idx]],
        covariates_partial_state    = [turing_data.covariates_partial_state[idx]],
        covariates_partial_regional = [turing_data.covariates_partial_regional[idx]],
        cases_start                 = turing_data.cases_start
    )
end

function select_states(turing_data, idxs)
    (
        num_states                  = length(idxs),
        num_impute                  = turing_data.num_impute,
        num_obs_states              = turing_data.num_obs_states[idxs],
        num_total_days              = turing_data.num_total_days,
        cases                       = turing_data.cases[idxs],
        deaths                      = turing_data.deaths[idxs],
        π                           = turing_data.π[idxs],
        π2                          = turing_data.π2[idxs],
        epidemic_start              = turing_data.epidemic_start[idxs],
        population                  = turing_data.population[idxs],
        serial_intervals            = turing_data.serial_intervals,
        week_index                  = turing_data.week_index[idxs],
        Region                      = turing_data.Region,
        num_weeks                   = turing_data.num_weeks,
        covariates                  = turing_data.covariates[idxs],
        covariates_partial_state    = turing_data.covariates_partial_state[idxs],
        covariates_partial_regional = turing_data.covariates_partial_regional[idxs],
        cases_start                 = turing_data.cases_start
    )
end

# turing_data = data.turing_data
turing_data = select_state(turing_data, 3);
# turing_data = select_state(turing_data, [1,2,3])
#---------------------------------------------------------------------------
# states     = data.states
# num_states = length(states)
model_def  = ImperialUSAcases.model
parameters = (
    warmup = 10,#parsed_args["num-warmup"],
    steps  = 10,#parsed_args["num-samples"],
    seed   = 1#parsed_args["seed"],
)
Random.seed!(parameters.seed);

m = model_def(
    turing_data.num_impute,
    turing_data.num_total_days,
    turing_data.cases,
    turing_data.deaths,
    turing_data.π,
    turing_data.π2,
    turing_data.covariates,
    turing_data.covariates_partial_regional,
    turing_data.covariates_partial_state,
    turing_data.epidemic_start,
    turing_data.population,
    turing_data.serial_intervals,
    turing_data.Region,
    turing_data.num_weeks,
    turing_data.week_index,
    turing_data.cases_start,
    false # <= DON'T predict
);
@info parameters
chain = sample(m, Prior(), 1000; progress=true)
chain = sample(m, NUTS(parameters.warmup, 0.95; max_depth=10), parameters.steps + parameters.warmup; progress=true)

@info "Saving at: $(projectdir("out", savename("chains_usa_cases", parameters, "jls")))"
safesave(projectdir("out", savename("chains_usa_cases", parameters, "jls")), chain)
