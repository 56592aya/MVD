using DataFrames
using DelimitedFiles
using ArgParse
using FileIO
using JLD2
using BenchmarkTools
using Logging
using Random
using Distributions
using LinearAlgebra
using Plots

include("AbstractCorpus.jl")
include("AbstractModel.jl")
include("AbstractSettings.jl")
include("AbstractCounts.jl")
include("AbstractDocument.jl")
include("utils2.jl")
include("dgp2.jl")
include("funcs2.jl")
