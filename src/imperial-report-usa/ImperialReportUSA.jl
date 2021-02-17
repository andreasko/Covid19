module ImperialReportUSA
using Turing, Distributions, StatsBase, ArgCheck

using ..Covid19: NegativeBinomial2

include("models.jl")
include("data.jl")
include("visualization.jl")

const model = model_v1 # <= defines the "official" model for this sub-module

end
