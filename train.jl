include("loader.jl")
const ε = 1e-12
function train(model, settings, folder, data_folder, h_map,count_params, mbs, nb, mb_size,perp1_list,perp2_list,VI_CONVERGED,
	hos1_dict,obs1_dict,hos2_dict,obs2_dict, mindex, epoch_count,_C1,_C2,_V1,_V2,_D,_D1,_D2,_K1,_K2,
	_terms1,_terms2,_counts1,_counts2,_lengths1,_lengths2,_train_ids, burnin)
	@info "VI Started"
	###
	complete_train_ids = [i for i in _train_ids if _lengths2[i] > 0]
	###
	for iter in 1:settings._MAX_VI_ITER
		# iter = 1
		check_eval = (epoch_count % settings._EVAL_EVERY == 0) || (epoch_count == 0)
		if mindex == (nb+1) || iter == 1
			if check_eval
				# model._α
				# if !burnin
				#
				# 	# update_α_newton_full!(model, complete_train_ids, settings)
				# end
				VI_CONVERGED = evaluate_at_epoch(folder,model, h_map, settings, count_params,hos1_dict,obs1_dict,hos2_dict,obs2_dict,
							perp1_list,perp2_list,VI_CONVERGED,epoch_count, _terms1, _terms2, _counts1, _counts2)

			end

			# mbs, nb = epoch_batches(_train_ids, mb_size)

			if burnin
				mbs, nb = epoch_batches(complete_train_ids, mb_size)
			else
				mbs, nb = epoch_batches(_train_ids, mb_size)
			end

			mindex = 1
		end
		check_epoch = (mindex  == nb)
		if check_epoch
			epoch_count += 1
			if epoch_count % settings._EVAL_EVERY == 0
				@info "i:$(iter) epoch :$(epoch_count)"
			end
		end
		mb = mbs[mindex]; _d1 = length(mb); _d2 = sum([_lengths2[d] != 0  for d in mb]);

		################################
			 ### Local Step ###
		################################
	 	init_γs!(model, mb)
		args = init_sstats!(model, settings, mb)
		# model._α[inds1, inds2] .= Truth_Params.Α
		update_local!(model, settings, mb,_terms1,_counts1,_terms2,_counts2, args...)
		# model._γ[complete_train_ids[1]]
		# model._γ[complete_train_ids[1]]
		################################
			  ### Global Step ###
		################################
		ρ = get_ρ(iter,settings)

		if burnin
			update_global!(model, ρ, length(complete_train_ids),length(complete_train_ids), _d1,_d2)
		else
			update_global!(model, ρ, _D1,_D2, _d1,_d2)
		end
		################################
			 ### Hparam Learning ###
		################################

		if burnin && (epoch_count > 40)
			@info "burnin turned off"
			burnin = false
		end

		if !burnin
			update_α_newton_full_one_shot!(model, _train_ids, settings)
		end
		# @load "$(data_folder)/truth" Truth_Params
		# inds1, inds2 = [2, 3, 5, 4, 10, 7, 1, 8, 6, 9],[1, 4, 2, 5, 3, 8, 7, 6, 9, 10]
		#
		# model._α[inds1, inds2] .= Truth_Params.Α
		# model._α_old = deepcopy(model._α)

		################################
		mindex += 1
		# iter += 1


		# inds1, inds2 = [4, 5, 2, 3, 1], [5, 4, 3, 1, 2]

		# @load "$(data_folder)/truth" Truth_Params
		# model._α[inds1, inds2] .= Truth_Params.Α
		# model._α .= rand(20, 20)
		################################
			###For FINAL Rounds###
		################################
		if iter == settings._MAX_VI_ITER || VI_CONVERGED
			@info "Final rounds"
			mb = collect(1:model._corpus1._D)[.!h_map]
			_d1 = length(mb)
			_d2 = length([d for d in mb if model._corpus2._docs[d]._length != 0]) ##func this
			update_local!(model, settings, mb)
			update_global!(model,1.0,count_params,_d1,_d2)
			break
		end
	end

	@save "$(folder)/model_at_last"  model
	@save "$(folder)/perp1_list"  perp1_list
	@save "$(folder)/perp2_list"  perp2_list
end
