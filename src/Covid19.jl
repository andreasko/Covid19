module Covid19

using DrWatson, Turing

export ImperialReport13,
    ImperialReportUSA,
    ImperialUSAcases,
    NegativeBinomial2,
    GammaMeanCv,
    generated_quantities,
    vectup2tupvec,
    arrarrarr2arr,
    plot_confidence_timeseries,
    plot_confidence_timeseries!

include("io.jl")
include("utils.jl")           # <= stuff that might also be included by sub-modules
include("visualization.jl")   # <= visualization stuff

# Different related reports
include("imperial-report13/ImperialReport13.jl")
include("imperial-report-usa/ImperialReportUSA.jl")
include("imperial-usa-cases/ImperialUSAcases.jl")
end # module
