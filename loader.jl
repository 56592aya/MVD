using Pkg
package_list = [:DataFrames,:DelimitedFiles,:ArgParse,:FileIO,:JLD2,:BenchmarkTools,:Logging,:Random,:Distributions,:LinearAlgebra,:Plots]
for p in package_list
    if in("$(p)",keys(Pkg.installed()))
        @eval using $(p)
    else
        Pkg.add("$(p)")
        @eval using $(p)
    end
end
include("AbstractCorpus.jl")
include("AbstractModel.jl")
include("AbstractSettings.jl")
include("AbstractCounts.jl")
include("AbstractDocument.jl")
include("utils.jl")
include("dgp.jl")
include("funcs.jl");
print("")
