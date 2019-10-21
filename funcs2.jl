
function update_Elogtheta!(Elog_, γ_)
	Elog_ .= Elog.(γ_)
end
function update_Elogtheta_i!(model, i)
	model.Elog_Theta[i] .= Elog(model.γ[i])
end

function update_Elogb!(model, mode)
	if mode == 1
		for (k,row) in enumerate(eachrow(model.b1))
			model.Elog_B1[k,:].= Elog(row)
		end
	else
		for (k,row) in enumerate(eachrow(model.b2))
			model.Elog_B2[k,:] .=  Elog(row)
		end
	end
end

function estimate_thetas(gamma)
	theta_est = deepcopy(gamma)
	for i in 1:length(theta_est)
		s = sum(gamma[i])
		theta_est[i] ./= s
	end
	return theta_est
end
function estimate_B(b_)
	res = zeros(Float64, size(b_))
	for k in 1:size(b_, 1)
		res[k,:] .= mean(Dirichlet(b_[k,:]))
	end
	return res
end

function optimize_γi!(model::MVD, i)
	@inbounds for I in eachindex(model.γ[i])
		model.γ[i][I] = model.Alpha[I] + model.sum_phi_1_i[I] + model.sum_phi_2_i[I]
	end
end
function optimize_b!(b_,len_mb, model_beta, sum_phi_mb,N)
	copyto!(b_, model_beta)
	@.(b_ += (N/len_mb) * sum_phi_mb)
end

function optimize_phi_iw!(model::MVD, i, mode::Int64, v::Int64)
	if mode == 1
		softmax!(model.temp, @.(model.Elog_Theta[i] + model.Elog_B1[:,v]))
	else
		softmax!(model.temp, @.(model.Elog_Theta[i] + model.Elog_B2[:,v]'))
	end
end

function update_phis_gammas!(model, i,zeroer_i,doc1,doc2,gamma_c)
	MAXLOOP = 500
	counter  = 0
	gamma_change = 500.0

	while !( gamma_c) && counter <= MAXLOOP
		# global counter, MAXLOOP,old_change, gamma_change
		copyto!(model.sum_phi_1_i, zeroer_i)
		copyto!(model.sum_phi_2_i, zeroer_i)
		for (w,val) in enumerate(doc1.terms)
			optimize_phi_iw!(model, i,1,val)
			@. model.sstat_i = doc1.counts[w] * model.temp
			@.(model.sum_phi_1_i += model.sstat_i)
		end

		for (w,val) in enumerate(doc2.terms)
			optimize_phi_iw!(model, i,2,val)
			@. model.sstat_i = doc2.counts[w] * model.temp
			@.(model.sum_phi_2_i += model.sstat_i)
		end
		optimize_γi!(model, i)
		update_Elogtheta_i!(model,i)
		gamma_change = mean_change(model.γ[i], model.old_γ)
		if (gamma_change < 1e-3) || counter == MAXLOOP
			gamma_c = true
			copyto!(model.sum_phi_1_i, zeroer_i)
			copyto!(model.sum_phi_2_i, zeroer_i)

			for (w,val) in enumerate(doc1.terms)
				optimize_phi_iw!(model, i,1,val)
				@. (model.sstat_i = doc1.counts[w] * model.temp)
				@.(model.sum_phi_1_i += model.sstat_i)
				model.sstat_mb_1 .= sum(model.sstat_i, dims = 2)[:,1]
				@.(model.sum_phi_1_mb[:,val] += model.sstat_mb_1)
				model.alpha_sstat[i] .+= model.sstat_i
			end

			for (w,val) in enumerate(doc2.terms)
				optimize_phi_iw!(model, i,2,val)
				@.(model.sstat_i = doc2.counts[w] * model.temp)
				@.(model.sum_phi_2_i += model.sstat_i)
				model.sstat_mb_2 .= sum(model.sstat_i , dims = 1)[1,:]
				@.(model.sum_phi_2_mb[:,val] += model.sstat_mb_2)
				model.alpha_sstat[i] .+= model.sstat_i
			end
			optimize_γi!(model, i)
			update_Elogtheta_i!(model,i)
		end
		copyto!(model.old_γ,model.γ[i])
		if counter == MAXLOOP
			gamma_c = true
		end
		counter += 1
	end
end
function calc_theta_bar_i(obs1_dict, obs2_dict, i, model, count_params,zeroer_i)
	update_Elogtheta_i!(model, i)
	doc1 = obs1_dict[i]
	doc2 = obs2_dict[i]
	corp1 = model.Corpus1.docs[i]
	corp2 = model.Corpus2.docs[i]
	copyto!(model.sum_phi_1_i, zeroer_i)
	copyto!(model.sum_phi_2_i, zeroer_i)
	copyto!(model.old_γ ,  model.γ[i])
	MAXLOOP = 200
	counter  = 0
	gamma_change = 500.0
	gamma_c = false
	model.γ[i] .= 1.0
	#############
	while !( gamma_c) && counter <= MAXLOOP
		copyto!(model.sum_phi_1_i, zeroer_i)
		obs_words_corp1inds = [find_all(d,corp1.terms)[1] for d in doc1]
		for (key,val) in enumerate(corp1.terms[obs_words_corp1inds])
			optimize_phi_iw!(model, i,1,val)
			@.(model.sstat_i = corp1.counts[key] * model.temp)
			@.(model.sum_phi_1_i += model.sstat_i)
		end
		copyto!(model.sum_phi_2_i, zeroer_i)
		obs_words_corp2inds = [find_all(d,corp2.terms)[1] for d in doc2]
		for (key,val) in enumerate(corp2.terms[obs_words_corp2inds])
			optimize_phi_iw!(model, i,2,val)
			@.(model.sstat_i = corp2.counts[key] * model.temp)
			@.(model.sum_phi_2_i += model.sstat_i)
		end
		optimize_γi!(model, i)
		update_Elogtheta_i!(model, i)
		gamma_change = mean_change(model.γ[i], model.old_γ)
		if (gamma_change < 1e-3) || counter == MAXLOOP
			gamma_c = true
		end
		copyto!(model.old_γ,model.γ[i])
		counter +=1
	end
	theta_bar = model.γ[i][:,:] ./ sum(model.γ[i])
	return theta_bar
end

function calc_perp(model,hos1_dict,obs1_dict,hos2_dict,obs2_dict,count_params, B1_est, B2_est,zeroer_i)
	corp1 = deepcopy(model.Corpus1)
	corp2 = deepcopy(model.Corpus2)
	l1 = 0.0
	l2 = 0.0

	for i in collect(keys(hos1_dict))
		theta_bar = calc_theta_bar_i(obs1_dict, obs2_dict,i, model, count_params,zeroer_i)
		for v in hos1_dict[i]
			tmp = 0.0
			for k in 1:count_params.K1
				tmp += ((B1_est[k,v]*sum(theta_bar, dims=2)[k,1]))
			end
			l1 += log(tmp)
		end
		for v in hos2_dict[i]
			tmp = 0.0
			for k in 1:count_params.K2
				tmp += ((B2_est[k,v]*sum(theta_bar, dims=1)[1,k]))
			end
			l2 += log(tmp)
		end

	end
	l1/= sum(length.(collect(values(hos1_dict))))
	l2/= sum(length.(collect(values(hos2_dict))))

	return exp(-l1), exp(-l2)
end

function update_alpha!(model, count_params)
	x = (sum(model.γ) - sum(model.alpha_sstat))/count_params.N
	copyto!(model.Alpha ,x)
end
# function update_alpha_trick!(model, mb, rate_, count_params)
# 	n = length(mb)
# 	K = prod(size(model.Alpha))
#
# 	init_ = deepcopy(model.Alpha)
# 	temp_ = deepcopy(init_)
# 	logphat = sum(hcat(vectorize_mat.([Elog(model.γ[i]) for i in mb])...), dims=2)[:,1] ./ (n)
# 	g = (count_params.N).*(vectorize_mat(-Elog(temp_)) .+ logphat)
# 	c = (count_params.N)*trigamma_(sum(temp_))
# 	q = (-count_params.N).*trigamma_.(vectorize_mat(temp_))
# 	b = sum(g./q)/(1.0/c+sum(1.0./q))
# 	dalpha = -(g .- b)./q
# 	temp_ .+= matricize_vec(rate_.*dalpha, model.K1, model.K2)
# 	if any(temp_ .<= 0.0) || any(isnan.(temp_))
# 		temp_ = (1.0/K).*mean(model.γ)
# 	end
# 	copyto!(model.Alpha, temp_)
# end
