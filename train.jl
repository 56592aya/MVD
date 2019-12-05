include("loader.jl")
function train(model, settings, folder, data_folder, h_map,count_params, mbs, nb, mb_size,perp1_list,perp2_list,VI_CONVERGED,
	hos1_dict,obs1_dict,hos2_dict,obs2_dict, mindex, epoch_count,_C1,_C2,_V1,_V2,_D,_D1,_D2,_K1,_K2,
	_terms1,_terms2,_counts1,_counts2,_lengths1,_lengths2,_train_ids)
	@info "VI Started"
	for iter in 1:settings._MAX_VI_ITER
		# iter = 1
		check_eval = (epoch_count % settings._EVAL_EVERY == 0) || (epoch_count == 0)
		check_epoch = (mindex  == nb)
		if mindex == (nb+1) || iter == 1
			if check_eval
				VI_CONVERGED = evaluate_at_epoch(folder,model, h_map, settings, count_params,hos1_dict,obs1_dict,hos2_dict,obs2_dict,
							perp1_list,perp2_list,VI_CONVERGED,epoch_count)
			end
			mbs, nb = epoch_batches(_train_ids, mb_size)
			mindex = 1
		end
		if check_epoch
			epoch_count += 1
			if epoch_count % settings._EVAL_EVERY == 0
				@info "i:$(iter) epoch :$(epoch_count)"
			end
		end
		mb = mbs[mindex]; _d1 = length(mb);_d2 = sum([_lengths2[d] != 0  for d in mb]);
		################################
			 ### Local Step ###
		################################
	 	init_γs!(model, mb)
		args = init_sstats!(model, settings, mb)
		update_local!(model, settings, mb,_terms1,_counts1,_terms2,_counts2, args...)
		################################
			  ### Global Step ###
		################################
		ρ = get_ρ(iter,settings)
		update_global!(model, ρ, _D1,_D2, _d1,_d2)
		################################
			 ### Hparam Learning ###
		################################
		update_α_newton_mb!(model,ρ, _D1,mb, h_map, settings)
		if maximum(model._α) > 0.5 && epoch_count > 15
			model._α .-= 0.5
			model._α[model._α .< 0.0] .= 1e-20
		end
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
