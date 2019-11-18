include("loader.jl")
Random.seed!(1234)
function main(args)
	s = ArgParseSettings()
    @add_arg_table s begin
		"--manual"
            help = "manual setting of alpha"
            action = "store_true"
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
		"--alpha"
			help = "alpha truth"
			arg_type = Float64
			default = .3
		"--eta1"
			help = "eta1 truth"
			arg_type = Float64
			default = .3
		"--eta2"
			help = "eta2 truth"
			arg_type = Float64
			default = .3
		"--constant"
			help = "alpha_sum"
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
	manual = parsed_args["manual"]
	K1 = parsed_args["k1"]
	K2 = parsed_args["k2"]
	V1 = parsed_args["v1"]
	V2 = parsed_args["v2"]
	α_single_truth = parsed_args["alpha"]
	η1_single_truth = parsed_args["eta1"]
	η2_single_truth = parsed_args["eta2"]
	wlen1_single = parsed_args["wlen1"]
	wlen2_single = parsed_args["wlen2"]
	c = parsed_args["constant"]


	# N = 10000
	# K1 = 5
	# K2 = 5
	# V1 = 100
	# V2 = 100
	# α_single_truth = 1.1
	# η1_single_truth = .05
	# η2_single_truth = .05
	# wlen1_single = 250
	# wlen2_single = 250
	# manual  = true
	# constant  = 1.0


	folder = mkdir("$(N)_$(K1)_$(K2)_$(V1)_$(V2)_$(α_single_truth)_$(η1_single_truth)_$(η2_single_truth)_$(manual)_$(c)")
	#########################
	α,Α, θ,Θ, ϕ1, ϕ2, η1, η2, V1, V2, corp1, corp2 =
	 Create_Truth(N, K1, K2, V1, V2,α_single_truth, η1_single_truth, η2_single_truth, wlen1_single, wlen2_single, manual,c)

	 α_truth,Α_truth, θ_truth,Θ_truth,ϕ1_truth, ϕ2_truth, η1_truth, η2_truth,V1, V2, corp1, corp2=
	 simulate_data(N, K1, K2, V1, V2,α_single_truth,η1_single_truth, η2_single_truth,wlen1_single, wlen2_single, manual,c)





	Truth_Params = Params(N,K1,K2,V1,V2,α_truth,Α_truth,θ_truth,Θ_truth,η1_truth,η2_truth,ϕ1_truth,ϕ2_truth)
	@save "$(folder)/truth" Truth_Params
	Docs1 = [Document(corp1[i], corp1[i], length(corp1[i])) for i in 1:length(corp1)]
	Docs2 = [Document(corp2[i], corp2[i], length(corp2[i])) for i in 1:length(corp2)]



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
	Plots.heatmap(Truth_Params.Α, yflip=true)
	savefig("$(folder)/Alpha_truth.png")
end

main(ARGS)
