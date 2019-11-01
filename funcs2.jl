
function update_Elogtheta!(Elog_, γ_)
	Elog_ .= Elog.(γ_)
end
function update_Elogtheta_i!(model::MVD, i::Int64)
	model.Elog_Theta[i] .= Elog(model.γ[i])
end
function update_Elogb!(model::MVD, mode::Int64)
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

function estimate_thetas(gamma::MatrixList{Float64})
	theta_est = deepcopy(gamma)
	for i in 1:length(theta_est)
		s = sum(gamma[i])
		theta_est[i] ./= s
	end
	return theta_est
end
function estimate_B(b_::Matrix{Float64})
	res = zeros(Float64, size(b_))
	for k in 1:size(b_, 1)
		res[k,:] .= mean(Dirichlet(b_[k,:]))
	end
	return res
end

function optimize_γi!(model::MVD, i::Int64)
	@inbounds for I in eachindex(model.γ[i])
		model.γ[i][I] = model.Alpha[I] + .5 + model.sum_phi_1_i[I] + model.sum_phi_2_i[I]
	end
end
function optimize_γi_perp!(model::MVD, i::Int64)
	@inbounds for I in eachindex(model.γ[i])
		model.γ[i][I] = model.Alpha[I]  + model.sum_phi_1_i[I] + model.sum_phi_2_i[I]
	end
end
function optimize_b!(b_::Matrix{Float64},len_mb::Int64, model_beta::Matrix{Float64}, sum_phi_mb::Matrix{Float64},N::Int64)
	copyto!(b_, model_beta)
	@.(b_ += .5 + (N/len_mb) * sum_phi_mb)
end

# function optimize_phi_iw!(model::MVD, i::Int64, mode::Int64, v::Int64)
# 	if mode == 1
# 	    softmax!(model.temp, @.(model.Elog_Theta[i] + model.Elog_B1[:,v]))
# 	else
# 		softmax!(model.temp, @.(model.Elog_Theta[i] + model.Elog_B2[:,v]'))
# 	end
# end

function optimize_phi_iw!(model::MVD, i::Int64, mode::Int64, v::Int64)
	if mode == 1
	    @. model.temp = exp(model.Elog_Theta[i] + model.Elog_B1[:,v])
		model.temp ./= sum(model.temp)
	else
		@. model.temp = exp(model.Elog_Theta[i] + model.Elog_B2[:,v]')
		model.temp ./= sum(model.temp)
	end
end
#back to copy
function update_phis_gammas!(model::MVD, i::Int64,settings::Settings,doc1::Document,doc2::Document,gamma_c::Bool)

	counter  = 0::Int64
	gamma_change = 500.0::Float64

	while !( gamma_c) && counter <= settings.MAX_GAMMA_ITER
		# global counter, MAXLOOP,old_change, gamma_change
		model.sum_phi_1_i .= settings.zeroer_i
		model.sum_phi_2_i .= settings.zeroer_i
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
		# println(gamma_change)
		# println(mean(abs.((model.γ[i] .- model.old_γ)/model.old_γ)))
		# model.old_γ .= model.γ[i]#remove
		if (gamma_change < settings.GAMMA_THRESHOLD) || counter == settings.MAX_GAMMA_ITER
			gamma_c = true
			model.sum_phi_1_i .= settings.zeroer_i
			model.sum_phi_2_i .= settings.zeroer_i

			for (w,val) in enumerate(doc1.terms)
				optimize_phi_iw!(model, i,1,val)
				@. (model.sstat_i = doc1.counts[w] * model.temp)
				@.(model.sum_phi_1_i += model.sstat_i)
				model.sstat_mb_1 .= sum(model.sstat_i, dims = 2)[:,1]
				@.(model.sum_phi_1_mb[:,val] += model.sstat_mb_1)
				# model.alpha_sstat[i] .+= model.sstat_i
			end

			for (w,val) in enumerate(doc2.terms)
				optimize_phi_iw!(model, i,2,val)
				@.(model.sstat_i = doc2.counts[w] * model.temp)
				@.(model.sum_phi_2_i += model.sstat_i)
				model.sstat_mb_2 .= sum(model.sstat_i , dims = 1)[1,:]
				@.(model.sum_phi_2_mb[:,val] += model.sstat_mb_2)
				# model.alpha_sstat[i] .+= model.sstat_i
			end
			optimize_γi!(model, i)
			update_Elogtheta_i!(model,i)
		end
		model.old_γ .= model.γ[i]

		if counter == settings.MAX_GAMMA_ITER
			gamma_c = true
		end
		counter += 1

	end
	# println(counter)
end
function calc_theta_bar_i(obs1_dict::Dict{Int64,Array{Int64,1}}, obs2_dict::Dict{Int64,Array{Int64,1}},
	 i::Int64, model::MVD, count_params::CountParams,settings::Settings)
	update_Elogtheta_i!(model, i)
	doc1 = obs1_dict[i]
	doc2 = obs2_dict[i]
	corp1 = model.Corpus1.docs[i]
	corp2 = model.Corpus2.docs[i]
	copyto!(model.sum_phi_1_i, settings.zeroer_i)
	copyto!(model.sum_phi_2_i, settings.zeroer_i)
	copyto!(model.old_γ ,  model.γ[i])

	counter  = 0
	gamma_change = 500.0
	gamma_c = false
	model.γ[i] .= 1.0
	#############
	while !( gamma_c) && counter <= settings.MAX_GAMMA_ITER
		copyto!(model.sum_phi_1_i, settings.zeroer_i)
		obs_words_corp1inds = [find_all(d,corp1.terms)[1] for d in doc1]
		for (key,val) in enumerate(corp1.terms[obs_words_corp1inds])
			optimize_phi_iw!(model, i,1,val)
			@.(model.sstat_i = corp1.counts[key] * model.temp)
			@.(model.sum_phi_1_i += model.sstat_i)
		end
		copyto!(model.sum_phi_2_i, settings.zeroer_i)
		obs_words_corp2inds = [find_all(d,corp2.terms)[1] for d in doc2]
		for (key,val) in enumerate(corp2.terms[obs_words_corp2inds])
			optimize_phi_iw!(model, i,2,val)
			@.(model.sstat_i = corp2.counts[key] * model.temp)
			@.(model.sum_phi_2_i += model.sstat_i)
		end
		optimize_γi_perp!(model, i)
		update_Elogtheta_i!(model, i)
		gamma_change = mean_change(model.γ[i], model.old_γ)
		# println(gamma_change)
		if (gamma_change < settings.GAMMA_THRESHOLD) || counter == settings.MAX_GAMMA_ITER
			gamma_c = true
		end
		copyto!(model.old_γ,model.γ[i])
		counter +=1
	end
	theta_bar = model.γ[i][:,:] ./ sum(model.γ[i])
	return theta_bar
end

function calc_perp(model::MVD,hos1_dict::Dict{Int64,Array{Int64,1}},
				   obs1_dict::Dict{Int64,Array{Int64,1}},hos2_dict::Dict{Int64,Array{Int64,1}},
				   obs2_dict::Dict{Int64,Array{Int64,1}},count_params::CountParams,
				   B1_est::Matrix{Float64}, B2_est::Matrix{Float64},settings::Settings)
	corp1 = deepcopy(model.Corpus1)
	corp2 = deepcopy(model.Corpus2)
	l1 = 0.0
	l2 = 0.0

	for i in collect(keys(hos1_dict))
		theta_bar = calc_theta_bar_i(obs1_dict, obs2_dict,i, model, count_params,settings)
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

function update_alpha_newton!(model::MVD, count_params::CountParams, h_map::Vector{Bool}, settings::Settings)
	N = count_params.N
	logphat = vectorize_mat(sum(Elog.(model.γ[.!h_map]))./N)
	counter = 0
	cond = true
	decay = 0
	K = prod(size(model.Alpha))
	Alpha = ones(Float64, 25)
	#Alpha = vectorize_mat(deepcopy(model.Alpha))
	Alpha_new = zeros(Float64,K)
	ga = zeros(Float64,K)
	Ha = zeros(Float64,K)

	while cond
		sumgh = 0.0
		sum1h = 0.0
		for k in 1:K
			#global sumgh, sum1h
			ga[k] = N*(digamma_(sum(Alpha))- digamma_(Alpha[k]) + logphat[k])
			Ha[k] = -N*trigamma_(Alpha[k])
			sumgh += ga[k]/Ha[k]
			sum1h += 1.0/Ha[k]
		end
		z = N*trigamma_(sum(Alpha))
		c = sumgh/(1.0/z + sum1h)
		while true

			singular = false
			for k in 1:K
				#global singular
				step_ = (settings.ALPHA_DECAY_FACTOR^decay) * (ga[k] - c)/Ha[k]
				if Alpha[k] <= step_
					singular = true
					break
				end
				Alpha_new[k] = Alpha[k] - step_
			end

			if singular
				decay += 1
				copyto!(Alpha_new,Alpha)
				if decay > settings.MAX_ALPHA_DECAY
					break
				end
			else
				break;
			end
		end
		cond = false
		mchange_ = mean_change(Alpha_new, Alpha)
		if mchange_ >= settings.ALPHA_THRESHOLD
			cond =true
		end
		if counter > settings.MAX_ALPHA_ITER
			cond = false
		end
		if decay > settings.MAX_ALPHA_DECAY
			break;
		end
		counter += 1
		copyto!(Alpha, Alpha_new)
	end
	copyto!(model.Alpha, matricize_vec(Alpha, model.K1, model.K2))

end

function update_alpha_newton!(model::MVD,ρ, count_params::CountParams,mb::Vector{Int64}, h_map::Vector{Bool}, settings::Settings)
	N = count_params.N
	n = length(mb)
	logphat = vectorize_mat(sum(Elog.(model.γ[mb]))./n)
	K = prod(size(model.Alpha))
	#Alpha = ones(Float64, 25)
	Alpha = vectorize_mat(deepcopy(model.Alpha))
	Alpha_new = zeros(Float64,K)
	ga = zeros(Float64,K)
	Ha = zeros(Float64,K)


	sumgh = 0.0
	sum1h = 0.0
	for k in 1:K
		#global sumgh, sum1h
		ga[k] = N*(digamma_(sum(Alpha))- digamma_(Alpha[k]) + logphat[k])
		Ha[k] = -N*trigamma_(Alpha[k])
		sumgh += ga[k]/Ha[k]
		sum1h += 1.0/Ha[k]
	end
	z = N*trigamma_(sum(Alpha))
	c = sumgh/(1.0/z + sum1h)

	step_ = ρ .* (ga .- c)./Ha
	if all(Alpha .> step_)
		Alpha_new = Alpha .- step_
		#copyto!(Alpha, Alpha_new)
		copyto!(model.Alpha, matricize_vec(Alpha_new, model.K1, model.K2))
	end
	# model.Alpha ./= sum(model.Alpha)
end

function update_beta1_newton!(model::MVD,ρ, settings::Settings)
	N = size(model.B1,1)
	logphat = (sum(Elog(model.b1[k,:]) for k in 1:N) ./ N)
	V = model.Corpus1.V
	#Alpha = ones(Float64, 25)
	B1 = deepcopy(model.B1[1,:])
	B1_new = zeros(Float64,V)
	ga = zeros(Float64,V)
	Ha = zeros(Float64,V)


	sumgh = 0.0
	sum1h = 0.0
	for v in 1:V
		#global sumgh, sum1h
		ga[v] = N*(digamma_(sum(B1))- digamma_(B1[v]) + logphat[v])
		Ha[v] = -N*trigamma_(B1[v])
		sumgh += ga[v]/Ha[v]
		sum1h += 1.0/Ha[v]
	end
	z = N*trigamma_(sum(B1))
	c = sumgh/(1.0/z + sum1h)

	step_ = ρ .* (ga .- c)./Ha
	if all(B1 .> step_)
		B1_new = B1 .- step_
		copyto!(model.B1, collect(repeat(B1_new, inner=(1,N))'))
	end
end
function update_beta2_newton!(model::MVD,ρ,  settings::Settings)
	N = size(model.B2,1)
	logphat = (sum(Elog(model.b2[k,:]) for k in 1:N) ./ N)
	V = model.Corpus2.V
	B2 = deepcopy(model.B2[1,:])
	B2_new = zeros(Float64,V)
	ga = zeros(Float64,V)
	Ha = zeros(Float64,V)


	sumgh = 0.0
	sum1h = 0.0
	for v in 1:V
		#global sumgh, sum1h
		ga[v] = N*(digamma_(sum(B2))- digamma_(B2[v]) + logphat[v])
		Ha[v] = -N*trigamma_(B2[v])
		sumgh += ga[v]/Ha[v]
		sum1h += 1.0/Ha[v]
	end
	z = N*trigamma_(sum(B2))
	c = sumgh/(1.0/z + sum1h)

	step_ = ρ .* (ga .- c)./Ha
	if all(B2 .> step_)
		B2_new = B2 .- step_
		copyto!(model.B2, collect(repeat(B2_new, inner=(1,N))'))
	end
end
