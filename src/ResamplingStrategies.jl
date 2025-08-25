module ResamplingStrategies

using MLJBase
using Random
using Statistics
using StatsBase: countmap, sample
using ScientificTypesBase

# Include strategies
include("main.jl")
export StratifiedHoldout

end # module ResamplingStrategies
