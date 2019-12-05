include("loader.jl")
Random.seed!(1234)
function main(args)
	s = ArgParseSettings()
    @add_arg_table s begin
		"--mode"
            help = "unidiagonal or bidiagonal"
			arg_type = String
            default = "uni"
		"-n"
            help = "number of docs"
            arg_type=Int64
            default=2000
		"--k1"
            help = "number of topics in mode 1"
            arg_type=Int64
            default=5
		"--k2"
            help = "number of topics in mode 2"
            arg_type=Int64
            default=5
		"--v1"               #number of communities
            help = "number of vocabs in mode 1"
            arg_type=Int64
            default=200
		"--v2"               #number of communities
            help = "number of vocabs in mode 2"
            arg_type=Int64
            default=200
		"--wlen1"
            help = "number of words per doc 1"
            arg_type=Int64
            default=200
		"--wlen2"
            help = "number of words per doc 2"
            arg_type=Int64
            default=200
		"--eta1"
			help = "eta1 truth"
			arg_type = Float64
			default = .3
		"--eta2"
			help = "eta2 truth"
			arg_type = Float64
			default = .3
		"-R"
			help = "weight on diag or shifted diag"
			arg_type = Float64
			default = .99
		"-s"
			help = "sum of alpha elements"
			arg_type = Float64
			default = 1.0

    end
    # # #
    parsed_args = ArgParse.parse_args(args,s) ##result is a Dict{String, Any}
    @info "Parsed args: "
    for (k,v) in parsed_args
        @info "  $k  =>  $(repr(v))"
    end
    @info "before parsing"

	N = parsed_args["n"]
	mode = parsed_args["mode"]
	K1 = parsed_args["k1"]
	K2 = parsed_args["k2"]
	V1 = parsed_args["v1"]
	V2 = parsed_args["v2"]
	η1_single_truth = parsed_args["eta1"]
	η2_single_truth = parsed_args["eta2"]
	wlen1_single = parsed_args["wlen1"]
	wlen2_single = parsed_args["wlen2"]
	R = parsed_args["R"]
	s_ = parsed_args["s"]

	# N = 10000
	# K1 = 10
	# K2 = 10
	# V1 = 50
	# V2 = 50
	# η1_single_truth = .05
	# η2_single_truth = .05
	# wlen1_single = 250
	# wlen2_single = 250
	# mode  = "uni"
	# R  = .99
	# s_ = 5.0

	if !isdir("Data")
		mkdir("Data")
	end
	folder = mkdir(joinpath("Data","$(N)_$(K1)_$(K2)_$(V1)_$(V2)_$(wlen1_single)_$(wlen2_single)_$(η1_single_truth)_$(η2_single_truth)_$(mode)_$(R)_$(s_)"))
	#########################
	α,Α, θ,Θ, ϕ1, ϕ2, η1, η2, V1, V2, corp1, corp2 =
	 Create_Truth(N, K1, K2, V1, V2, η1_single_truth, η2_single_truth, wlen1_single, wlen2_single, R,mode,s_)

	 α_truth,Α_truth, θ_truth,Θ_truth,ϕ1_truth, ϕ2_truth, η1_truth, η2_truth,V1, V2, corp1, corp2=
	 simulate_data(N, K1, K2, V1, V2,η1_single_truth, η2_single_truth,wlen1_single, wlen2_single, R,mode,s_)





	Truth_Params = Params(N,K1,K2,V1,V2,α_truth,Α_truth,θ_truth,Θ_truth,η1_truth,η2_truth,ϕ1_truth,ϕ2_truth)
	@save "$(folder)/truth" Truth_Params
	Docs1 = [Document(corp1[i], corp1[i], length(corp1[i])) for i in 1:length(corp1)]
	Docs2 = [Document(corp2[i], corp2[i], length(corp2[i])) for i in 1:length(corp2)]
	@info "Data generated\n"


	Corpus1 = Corpus(Docs1, length(Docs1), V1)
	Corpus2 = Corpus(Docs2, length(Docs2), V2)
	@save "$(folder)/corpus1" Corpus1
	@save "$(folder)/corpus2" Corpus2

	ϕ11,chert1 = sort_by_argmax!(deepcopy(collect(transpose(Truth_Params.ϕ1))))
	ϕ22,chert2 = sort_by_argmax!(deepcopy(collect(transpose(Truth_Params.ϕ2))))
	Plots.heatmap(ϕ11, yflip=true)
	savefig("$(folder)/phi1_truth.png")
	Plots.heatmap(ϕ22, yflip=true)
	savefig("$(folder)/phi2_truth.png")
	Θ00, chert0 = sort_by_argmax!(deepcopy(Truth_Params.Θ_vec))
	Θ1 = [sum(Truth_Params.Θ[i], dims=2)[:,1] for i in 1:N]
	Θ1 = collect((hcat(Θ1...)'))
	Θ11,chert1 = sort_by_argmax!(deepcopy(Θ1))
	Θ2 = [sum(Truth_Params.Θ[i], dims=1)[1,:] for i in 1:N]
	Θ2 = collect((hcat(Θ2...)'))
	Θ22,chert2 = sort_by_argmax!(deepcopy(Θ2))
	Plots.heatmap(Θ00, yflip=true)
	savefig("$(folder)/Thetavec_truth.png")
	Plots.heatmap(Θ11, yflip=true)
	savefig("$(folder)/Theta1_truth.png")
	Plots.heatmap(Θ22, yflip=true)
	savefig("$(folder)/Theta2_truth.png")

	Plots.heatmap(Truth_Params.Α, yflip=true)
	savefig("$(folder)/Alpha_truth.png")
end

main(ARGS)
print("");
