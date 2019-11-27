function optimize_γi!(model::MVD, i::Int64)
	@inbounds for I in eachindex(model.γ[i])
		model.γ[i][I] = model.α[I] + .5 + model.sum_π_1_i[I] + model.sum_π_2_i[I]
	end
end
function optimize_γi_perp!(model::MVD, i::Int64)
	@inbounds for I in eachindex(model.γ[i])
		model.γ[i][I] = model.α[I]  + model.sum_π_1_i[I] + model.sum_π_2_i[I]
	end
end
function optimize_λ!(lambda_::Matrix{Float64},len_mb::Int64, model_eta::Matrix{Float64}, sum_π_mb::Matrix{Float64},N::Int64)
	copyto!(lambda_, model_eta)
	#@.(lambda_ +=  (N/len_mb) * sum_π_mb)
	@.(lambda_ += 0.5 + (N/len_mb) * sum_π_mb)
end
function optimize_π_iw!(model::MVD, i::Int64, mode::Int64, v::Int64)
	if mode == 1
	    @. model.π_temp = exp(model.Elog_Θ[i] + model.Elog_ϕ1[:,v])+1e-100
		model.π_temp ./= sum(model.π_temp)
	else
		@. model.π_temp = exp(model.Elog_Θ[i] + model.Elog_ϕ2[:,v]') + 1e-100
		model.π_temp ./= sum(model.π_temp)
	end
end

function update_local!(model::MVD, i::Int64,settings::Settings,doc1::Document,doc2::Document,gamma_flag::Bool)

	counter  = 0::Int64
	gamma_change = 500.0::Float64
	while !( gamma_flag) && counter <= settings.MAX_GAMMA_ITER
		model.sum_π_1_i .= settings.zeroer_i
		model.sum_π_2_i .= settings.zeroer_i
		for (w,val) in enumerate(doc1.terms)
			optimize_π_iw!(model, i,1,val)
			@. model.sstat_i = doc1.counts[w] * model.π_temp
			@.(model.sum_π_1_i += model.sstat_i)
		end

		for (w,val) in enumerate(doc2.terms)
			optimize_π_iw!(model, i,2,val)
			@. model.sstat_i = doc2.counts[w] * model.π_temp
			@.(model.sum_π_2_i += model.sstat_i)
		end
		if doc2.len > 0
			optimize_γi_perp!(model, i)
		else ## or possibly if K >5
			optimize_γi_perp!(model, i)
		end
		update_ElogΘ_i!(model,i)
		gamma_change = mean_change(model.γ[i], model.old_γ)
		if (gamma_change < settings.GAMMA_THRESHOLD) || counter == settings.MAX_GAMMA_ITER
			gamma_flag = true
			model.sum_π_1_i .= settings.zeroer_i
			model.sum_π_2_i .= settings.zeroer_i

			for (w,val) in enumerate(doc1.terms)
				optimize_π_iw!(model, i,1,val)
				@. (model.sstat_i = doc1.counts[w] * model.π_temp)
				@.(model.sum_π_1_i += model.sstat_i)
				model.sstat_mb_1 .= sum(model.sstat_i, dims = 2)[:,1]
				@.(model.sum_π_1_mb[:,val] += model.sstat_mb_1)
			end

			for (w,val) in enumerate(doc2.terms)
				optimize_π_iw!(model, i,2,val)
				@.(model.sstat_i = doc2.counts[w] * model.π_temp)
				@.(model.sum_π_2_i += model.sstat_i)
				model.sstat_mb_2 .= sum(model.sstat_i , dims = 1)[1,:]
				@.(model.sum_π_2_mb[:,val] += model.sstat_mb_2)
			end
			if doc2.len > 0
				optimize_γi!(model, i)
			else
				optimize_γi_perp!(model, i)
			end
			update_ElogΘ_i!(model,i)
		end
		model.old_γ .= model.γ[i]

		if counter == settings.MAX_GAMMA_ITER
			gamma_flag = true
		end
		counter += 1

	end
end
