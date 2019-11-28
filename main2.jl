include("loader2.jl")
Random.seed!(1234)

function train(model, settings, folder, data_folder, h_map,count_params, mbs, nb, mb_size,perp1_list,perp2_list,VI_CONVERGED,
	hos1_dict,obs1_dict,hos2_dict,obs2_dict, mindex, epoch_count,_C1,_C2,_V1,_V2,_D,_D1,_D2,_K1,_K2,
	_terms1,_terms2,_counts1,_counts2,_lengths1,_lengths2,_train_ids)


	@info "VI Started"

	for iter in 1:settings._MAX_VI_ITER
		# iter = 1
		if mindex == (nb+1) || iter == 1
			mbs, nb = epoch_batches(_train_ids, mb_size, h_map)
			mindex = 1
			if (epoch_count % settings._EVAL_EVERY == 0) || (epoch_count == 0)
				ϕ1_est = estimate_ϕ(model._λ1)
				ϕ2_est = estimate_ϕ(model._λ2)
				@info "starting to calc perp"
				p1, p2 = calc_perp(model,hos1_dict,obs1_dict,hos2_dict,obs2_dict,
				count_params, ϕ1_est, ϕ2_est, settings)
				perp1_list = vcat(perp1_list, p1)
				@info "perp1=$(p1)"
				perp2_list = vcat(perp2_list, p2)
				@info "perp2=$(p2)"
				@save "$(folder)/perp1_at_$(epoch_count)"  perp1_list
				@save "$(folder)/perp2_at_$(epoch_count)"  perp2_list
				@save "$(folder)/model_at_epoch_$(epoch_count)"  model

				if length(perp1_list) > 2
					if (abs(perp1_list[end]-perp1_list[end-1])/perp1_list[end] < settings._VI_THRESHOLD) &&
						(abs(perp2_list[end]-perp2_list[end-1])/perp2_list[end] < settings._VI_THRESHOLD)
						VI_CONVERGED  = true
					end
				end
			end
		end
		if mindex  == nb
			epoch_count += 1
			if epoch_count % settings._EVAL_EVERY == 0
				@info "i:$(iter) epoch :$(epoch_count)"

			end
		end
		################################
			 ### Local Step ###
		################################
		mb = mbs[mindex]
		len_mb2 = sum([_lengths2[d] != 0  for d in mb]) ##func this

		ρ = get_ρ(iter,settings)
	 	init_γs!(model, mb)
		init_sstats!(model, settings)
		update_local!(model, settings, mb,_C1,_C2,_terms1,_counts1,_terms2,_counts2)
		################################
			  ### Global Step ###
		################################
		update_global!(model, ρ, _D1,_D2, mb,len_mb2)
		################################
			 ### Hparam Learning ###
		################################
		update_α_newton_mb!(model,ρ, _D1,mb, h_map, settings)
		update_η1_newton_mb!(model,ρ, settings)
		update_η2_newton_mb!(model,ρ, settings)
		################################
		mindex += 1
		# iter += 1

		################################
			###For FINAL Rounds###
		################################
		if iter == settings._MAX_VI_ITER || VI_CONVERGED
			@info "Final rounds"
			mb = collect(1:model._corpus1._D)[.!h_map]
			len_mb2 = length([d for d in mb if model._corpus2._docs[d]._length != 0]) ##func this
			update_local!(model, settings, mb)
			update_global!(model,1.0,count_params,mb,len_mb2)
			break
		end
	end

	@save "$(folder)/model_at_last"  model
	@save "$(folder)/perp1_list"  perp1_list
	@save "$(folder)/perp2_list"  perp2_list
end

function main(args)
	s = ArgParseSettings()
    @add_arg_table s begin
		"--data"            #data folder
            arg_type = String
            required = true
		"--all"               #if sparsity for all
            help = "If sparsity for all"
            action = "store_true"
		"--sparsity"               #sparsity
            help = "percent not available"
            arg_type=Float64
            default=.5
		"--k1"               #number of communities
            help = "number of topics in mode 1"
            arg_type=Int64
            default=5
		"--k2"               #number of communities
            help = "number of topics in mode 2"
            arg_type=Int64
            default=5
        "--mbsize"
            help = "number of docs in a minibatch"
            arg_type=Int64
            default=64
        "--maxiter"
            help = "maximum number of iterations"
            arg_type=Int64
            default=5000
        "--every"
            help = "eval every number of iterations"
            arg_type=Int64
            default=10
		"--kappa"
			help = "kappa for learning rate"
			arg_type = Float64
			default = .5
		"--alpha_prior"
			help = "alpha prior"
			arg_type = Float64
			default = .3
		"--eta1_prior"
			help = "eta1 prior"
			arg_type = Float64
			default = .3
		"--eta2_prior"
			help = "eta2 prior"
			arg_type = Float64
			default = .3
		"-S"
			help = "S for learning rate"
			arg_type = Float64
			default = 256.0
		"--holdout"
			help = "holdout"
			arg_type = Float64
			default = .01
    end
    # # #
    parsed_args = ArgParse.parse_args(args,s) ##result is a Dict{String, Any}
    @info "Parsed args: "
    for (k,v) in parsed_args
        @info "  $k  =>  $(repr(v))"
    end
    @info "before parsing"
	data_folder = joinpath("Data",parsed_args["data"])
	K1 = parsed_args["k1"]
	K2 = parsed_args["k2"]
	α_single_prior = parsed_args["alpha_prior"]
	η1_single_prior = parsed_args["eta1_prior"]
	η2_single_prior = parsed_args["eta2_prior"]
	S = parsed_args["S"]
	κ = parsed_args["kappa"]
	every = parsed_args["every"]
	MAXITER = parsed_args["maxiter"]
	mb_size = parsed_args["mbsize"]
	h = parsed_args["holdout"]
	all_ = parsed_args["all"]
	sparsity = parsed_args["sparsity"]
	# global K1 = 5
	# global K2 = 5
	# global α_single_prior = .04
	# global η1_single_prior = .02
	# global η2_single_prior = .02
	# global S = 256.0
	# global κ = .7
	# global every = 1
	# global MAXITER = 80000
	# global mb_size = 256
	# global h = 0.005
	# global data_folder = joinpath("Data","10000_5_5_50_50_0.25_0.25_uni_0.9_1.0")
	# global all_ = true
	# global sparsity = 0.0
	folder = mkdir(joinpath(data_folder,"est_$(K1)_$(K2)_$(mb_size)_$(MAXITER)_$(h)_$(S)_$(κ)_$(every)_$(α_single_prior)_$(η1_single_prior)_$(η2_single_prior)_$(all_)_$(sparsity)"))
	cp("funcs.jl","$(folder)/funcs_used.jl")
	cp("main.jl","$(folder)/main_used.jl")
	@load "$(data_folder)/corpus1" Corpus1
	@load "$(data_folder)/corpus2" Corpus2
	D = max(Corpus1._D, Corpus2._D)
	model = MVD(K1, K2, Corpus1, Corpus2, α_single_prior,η1_single_prior,η2_single_prior)
	fix_corp!(model, folder)
	figure_sparsity!(model,sparsity,all_, folder)
	h_map = setup_hmap(model, h,D)
	@save "$(folder)/h_map" h_map
	mbs, nb = epoch_batches(collect(1:D)[.!h_map], mb_size, h_map)
	mindex, epoch_count = 1,0
	hos1_dict,obs1_dict,hos2_dict,obs2_dict =split_ho_obs(model, h_map)
	D2 = sum([1 for i in collect(1:model._corpus1._D)[.!h_map] if model._corpus2._docs[i]._length != 0])
	count_params = TrainCounts(model._corpus1._D-sum(h_map),D2, model._K1, model._K2)

	dir_expectationByRow!(model._elogϕ1, model._λ1)
	dir_expectationByRow!(model._elogϕ2, model._λ2)
	VI_CONVERGED = false
	perp1_list = Float64[]
	perp2_list = Float64[]
	MAX_VI_ITER = MAXITER
	MAX_ALPHA_ITER = 1000
	MAX_GAMMA_ITER = 1000
	MAX_ALPHA_DECAY= 10
	ALPHA_DECAY_FACTOR = .8
	ALPHA_THRESHOLD = 1e-5
	GAMMA_THRESHOLD =1e-3
	VI_THRESHOLD = 1e-8
	EVAL_EVERY = every
	LR_OFFSET, LR_KAPPA = S, κ
	settings = Settings(model._K1, model._K2, model._corpus1, model._corpus2,
	MAX_VI_ITER,MAX_ALPHA_ITER,MAX_GAMMA_ITER,MAX_ALPHA_DECAY,
	ALPHA_DECAY_FACTOR,ALPHA_THRESHOLD,GAMMA_THRESHOLD,VI_THRESHOLD,
	EVAL_EVERY, LR_OFFSET, LR_KAPPA)
	_C1 = model._corpus1._docs
	_C2 = model._corpus2._docs
	_V1 = model._corpus1._V
	_V2 = model._corpus2._V
	_D = model._corpus1._D
	_D1 = count_params._D1
	_D2 = count_params._D2
	_K1 = count_params._K1
	_K2 = count_params._K2
	_terms1 = [_C1[i]._terms for i in 1:model._corpus1._D]
	_terms2 = [_C2[i]._terms for i in 1:model._corpus2._D]
	_counts1 = [_C1[i]._counts for i in 1:model._corpus1._D]
	_counts2 = [_C2[i]._counts for i in 1:model._corpus2._D]
	_lengths1 = [_C1[i]._length for i in 1:model._corpus1._D]
	_lengths2 = [_C2[i]._length for i in 1:model._corpus2._D]
	_train_ids = collect(1:_D)[.!h_map]

	train(model, settings, folder, data_folder, h_map,count_params, mbs, nb, mb_size,perp1_list,perp2_list,VI_CONVERGED,
		hos1_dict,obs1_dict,hos2_dict,obs2_dict, mindex, epoch_count,_C1,_C2,_V1,_V2,_D,_D1,_D2,_K1,_K2,_terms1,_terms2,_counts1,_counts2,_lengths1,_lengths2,_train_ids)
end

main(ARGS)
