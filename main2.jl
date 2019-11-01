include("loader.jl")
Random.seed!(1234)

function train(model, settings, folder, data_folder, h_map,count_params, mbs, nb, mb_size,perp1_list,perp2_list,VI_CONVERGED,
	hos1_dict,obs1_dict,hos2_dict,obs2_dict, mindex, epoch_count)
	@info "VI Started"
	for iter in 1:settings.MAX_VI_ITER
		# iter = 1
		# global model, mindex, nb, mbs, count_params,mb_size, perp1_list, perp2_list,epoch_count,settings, VI_CONVERGED, h_map,hos1_dict,obs1_dict,hos2_dict,obs2_dict
		if mindex == (nb+1) || iter == 1

			mbs, nb = epoch_batches(model.Corpus1.N, mb_size, h_map)
			mindex = 1

			if (epoch_count % settings.EVAL_EVERY == 0) || (epoch_count == 0)
				x1 = deepcopy(model.b1)
				for I in CartesianIndices(x1)
					if x1[I]-.5 > 0
						x1[I] -= .5
					end
				end
				x2 = deepcopy(model.b2)
				for I in CartesianIndices(x2)
					if x2[I]-.5 > 0
						x2[I] -= .5
					end
				end
				B1_est = estimate_B(x1)
				B2_est = estimate_B(x2)
				@info "starting to calc perp"
				p1, p2 = calc_perp(model,hos1_dict,obs1_dict,hos2_dict,obs2_dict,
				count_params, B1_est, B2_est,settings)
				perp1_list = vcat(perp1_list, p1)
				@info "perp1=$(p1)"
				perp2_list = vcat(perp2_list, p2)
				@info "perp2=$(p2)"
				@save "$(folder)/perp1_at_$(epoch_count)"  perp1_list
				@save "$(folder)/perp2_at_$(epoch_count)"  perp2_list
				@save "$(folder)/model_at_epoch_$(epoch_count)"  model

				if length(perp1_list) > 2
					if (abs(perp1_list[end]-perp1_list[end-1])/perp1_list[end] < settings.VI_THRESHOLD) &&
						(abs(perp2_list[end]-perp2_list[end-1])/perp2_list[end] < settings.VI_THRESHOLD)
						VI_CONVERGED  = true
					end
				end
			end
		end

		if mindex  == nb
			epoch_count += 1
			if epoch_count % settings.EVAL_EVERY == 0
				@info "i:$(iter) epoch :$(epoch_count)"

			end
		end

		mb = mbs[mindex]
		len_mb2 = length([i for i in mb if model.Corpus2.docs[i].len != 0]) ##func this
		ρ = get_lr(iter,settings)
		#ρ = get_lr(epoch_count, mb,mindex,settings)
		################################
			 ### Local Step ###
		################################
		for i in mb

			# model.γ[i] .= 1.0
			model.γ[i] = rand(Gamma(100.0, 0.01), (model.K1,model.K2))
		end
		copyto!(model.sum_phi_1_mb, settings.zeroer_mb_1)
		copyto!(model.sum_phi_2_mb, settings.zeroer_mb_2)
		copyto!(model.sum_phi_1_i,  settings.zeroer_i)
		copyto!(model.sum_phi_2_i, settings.zeroer_i)

		# if epoch_count < 10
		# 	settings.MAX_GAMMA_ITER = 50
		# else
		# 	settings.MAX_GAMMA_ITER = 50
		# end

		for i in mb

			update_Elogtheta_i!(model, i)
			doc1 = model.Corpus1.docs[i]
			doc2 = model.Corpus2.docs[i]
			copyto!(model.old_γ, model.γ[i])
			gamma_c = false
			update_phis_gammas!(model, i,settings,doc1,doc2,gamma_c)
		end
		################################
			  ### Global Step ###
		################################
		copyto!(model.old_b1,  model.b1)
		optimize_b!(model.b1, length(mb), model.B1, model.sum_phi_1_mb, count_params.N)
		model.b1 .= (1.0-ρ).*model.old_b1 .+ ρ.*model.b1
		update_Elogb!(model, 1)
		copyto!(model.old_b2,model.b2)
		optimize_b!(model.b2,len_mb2, model.B2, model.sum_phi_2_mb,count_params.N2)
		model.b2 .= (1.0-ρ).*model.old_b2 .+ ρ.*model.b2
		update_Elogb!(model, 2)
		################################
			 ### Hparam Learning ###
		################################
		if epoch_count < 2
			nothing;
		elseif epoch_count>=2 && epoch_count< 5
			copyto!(model.old_Alpha,model.Alpha)
			update_alpha_newton!(model, count_params, h_map, settings)
			model.Alpha .= (1.0-ρ).*model.old_Alpha .+ ρ.*model.Alpha
		else
			update_alpha_newton!(model,ρ, count_params,mb, h_map, settings)
			update_beta1_newton!(model,ρ, settings)
			update_beta2_newton!(model,ρ, settings)
		end
		################################

		# println(mindex == nb)
		mindex += 1
		#iter += 1
		################################
			###For FINAL Rounds###
		################################
		if iter == settings.MAX_VI_ITER || VI_CONVERGED
			@info "Final rounds"
			mb = collect(1:count_params.N)[.!h_map]
			for i in mb
				model.γ[i] .= 1.0
			end
			copyto!(model.sum_phi_1_mb, settings.zeroer_mb_1)
			copyto!(model.sum_phi_2_mb, settings.zeroer_mb_1)
			copyto!(model.sum_phi_1_i,  settings.zeroer_i)
			copyto!(model.sum_phi_2_i, settings.zeroer_i)
			for i in mb
				update_Elogtheta_i!(model, i)
				doc1 = model.Corpus1.docs[i]
				doc2 = model.Corpus2.docs[i]
				copyto!(model.old_γ, model.γ[i])
				gamma_c = false
				update_phis_gammas!(model, i,settings,doc1,doc2,gamma_c)
			end
			optimize_b!(model.b1, length(mb), model.B1, model.sum_phi_1_mb, count_params.N)
			update_Elogb!(model, 1)
			optimize_b!(model.b2,len_mb2, model.B2, model.sum_phi_2_mb,count_params.N2)
			update_Elogb!(model, 2)
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
		"--beta1_prior"
			help = "beta1 prior"
			arg_type = Float64
			default = .3
		"--beta2_prior"
			help = "beta2 prior"
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

	data_folder = parsed_args["data"]
	K1 = parsed_args["k1"]
	K2 = parsed_args["k2"]
	α_single_prior = parsed_args["alpha_prior"]
	β1_single_prior = parsed_args["beta1_prior"]
	β2_single_prior = parsed_args["beta2_prior"]
	S = parsed_args["S"]
	κ = parsed_args["kappa"]
	every = parsed_args["every"]
	MAXITER = parsed_args["maxiter"]
	mb_size = parsed_args["mbsize"]
	h = parsed_args["holdout"]
	all_ = parsed_args["all"]
	sparsity = parsed_args["sparsity"]
	#global K1 = 5
	#global K2 = 5
	#global α_single_prior = .5
	#global β1_single_prior = .5
	#global β2_single_prior = .5
	#global S = 256.0
	#global κ = .6
	#global every = 1
	#global MAXITER = 80000
	#global mb_size = 256
	#global h = 0.005
	#global data_folder = "10000_5_5_50_50_0.5_0.2_0.2_true"
	#global all_ = true
	#global sparsity = 0.0
	folder = mkdir(joinpath(data_folder,"est_$(K1)_$(K2)_$(mb_size)_$(MAXITER)_$(h)_$(S)_$(κ)_$(every)_$(α_single_prior)_$(β1_single_prior)_$(β2_single_prior)_$(all_)_$(sparsity)"))
	@load "$(data_folder)/corpus1" Corpus1
	@load "$(data_folder)/corpus2" Corpus2
	N = max(Corpus1.N, Corpus2.N)
	model = MVD(K1, K2, Corpus1, Corpus2, α_single_prior,β1_single_prior,β2_single_prior)
	fix_corp!(model, folder)
	figure_sparsity!(model,sparsity,all_, folder)
	h_map = setup_hmap(model, h,N)
	@save "$(folder)/h_map" h_map
	mbs, nb = epoch_batches(N, mb_size, h_map)
	mindex, epoch_count = 1,0
	hos1_dict,obs1_dict,hos2_dict,obs2_dict =split_ho_obs(model, h_map)
	N2 = sum([1 for i in collect(1:model.Corpus1.N)[.!h_map] if model.Corpus2.docs[i].len != 0])
	count_params = CountParams(model.Corpus1.N-sum(h_map),N2, model.K1, model.K2)
	update_Elogtheta!(model.Elog_Theta, model.γ)
	update_Elogb!(model,1)
	update_Elogb!(model,2)
	VI_CONVERGED = false
	perp1_list = Float64[]
	perp2_list = Float64[]
	MAX_VI_ITER = MAXITER
	MAX_ALPHA_ITER = 1000
	MAX_GAMMA_ITER = 350
	MAX_ALPHA_DECAY= 10
	ALPHA_DECAY_FACTOR = .8
	ALPHA_THRESHOLD = 1e-5
	GAMMA_THRESHOLD =1e-3
	VI_THRESHOLD = 1e-8
	EVAL_EVERY = every
	LR_OFFSET, LR_KAPPA = S, κ
	settings = Settings(model.K1, model.K2, model.Corpus1, model.Corpus2,
	MAX_VI_ITER,MAX_ALPHA_ITER,MAX_GAMMA_ITER,MAX_ALPHA_DECAY,
	ALPHA_DECAY_FACTOR,ALPHA_THRESHOLD,GAMMA_THRESHOLD,VI_THRESHOLD,
	EVAL_EVERY, LR_OFFSET, LR_KAPPA)

	train(model, settings, folder, data_folder, h_map,count_params, mbs, nb, mb_size,perp1_list,perp2_list,VI_CONVERGED,
		hos1_dict,obs1_dict,hos2_dict,obs2_dict, mindex, epoch_count)
end

main(ARGS)
