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
using Random, Dates, Turing, Bijectors, DrWatson

using Base.Threads
nthreads()
#----------------------------------------------------------------------------
data = ImperialReportUSA.load_data(datadir("imperial-report-usa", "processed.rds"))

states = data.states
num_states = length(states)

# STUFF
turing_data = data.turing_data;
#----------------------------------------------------------------------------
# turing_data = data.turing_data
# i = 3
# turing_data = (
#     num_states = 1,
#     num_impute = turing_data.num_impute,
#     num_obs_states = [turing_data.num_obs_states[i]],
#     num_total_days = turing_data.num_total_days,
#     cases          = [turing_data.cases[i]],
#     deaths         = [turing_data.deaths[i]],
#     π              = [turing_data.π[i]],
#     epidemic_start = [turing_data.epidemic_start[i]],
#     population     = [turing_data.population[i]],
#     serial_intervals = turing_data.serial_intervals,
#     week_index       = [turing_data.week_index[i]],
#     Region           = turing_data.Region,
#     num_weeks = turing_data.num_weeks,
#     covariates = [turing_data.covariates[i]],
#     covariates_partial_state = [turing_data.covariates_partial_state[i]],
#     covariates_partial_regional = [turing_data.covariates_partial_regional[i]],
# );

model_def = ImperialReportUSA.model;
#---------------------------------------------------------------------------
parameters = (
    warmup = parsed_args["num-warmup"],
    steps = parsed_args["num-samples"],
    seed = parsed_args["seed"],
)
Random.seed!(parameters.seed);

m = model_def(
    turing_data.num_impute,
    turing_data.num_total_days,
    turing_data.cases,
    turing_data.deaths,
    turing_data.π,
    turing_data.covariates,
    turing_data.covariates_partial_regional,
    turing_data.covariates_partial_state,
    turing_data.epidemic_start,
    turing_data.population,
    turing_data.serial_intervals,
    turing_data.Region,
    turing_data.num_weeks,
    turing_data.week_index,
    false # <= DON'T predict
);

@info parameters
chain = sample(m, NUTS(parameters.warmup, 0.95; max_depth=10), parameters.steps + parameters.warmup; progress=true)

@info "Saving at: $(projectdir("out", savename("chains", parameters, "jls")))"
safesave(projectdir("out", savename("chains", parameters, "jls")), chain)
