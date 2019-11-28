function estimate_Θs(gamma)
	theta_est = deepcopy(gamma)
	for i in 1:length(theta_est)
		s = sum(gamma[i])
		theta_est[i] ./= s
	end
	return theta_est
end
function estimate_ϕ(lambda_::Matrix{Float64})
	res = zeros(Float64, size(lambda_))
	for k in 1:size(lambda_, 1)
		res[k,:] .= mean(Dirichlet(lambda_[k,:]))
	end
	return res
end

function optimize_γd!(model, γd,sumπ1_i,sumπ2_i)
	@. (γd =  model._α + sumπ1_i + sumπ2_i + .5)
end
function optimize_γd_perp!(model, γd,sumπ1_i,sumπ2_i)
	@. γd =  model._α + sumπ1_i + sumπ2_i
end
function optimize_λ!(lambda_::Matrix{Float64},len_mb::Int64, model_eta::Matrix{Float64}, sum_π_mb::Matrix{Float64},D::Int64)
	copyto!(lambda_, model_eta)
	@.(lambda_ += 0.5 + (D/len_mb) * sum_π_mb)
end
function init_γs!(model::MVD, mb::Vector{Int64})
	s = ones(Float64, (model._K1,model._K2))
	for i in mb
		model._γ[i] .= s
	end
end
function init_sstats!(model::MVD, settings::Settings)
	copyto!(model._sumπ1_mb, settings._zeroer_K1V1)
	copyto!(model._sumπ2_mb, settings._zeroer_K2V2)
end
function optimize_π_iw!(πtemp, expelogΘ, expelogϕ)
	@. (πtemp = (expelogΘ * expelogϕ) + 1e-100)
	πtemp ./= sum(πtemp)
end


function update_local!(model::MVD,settings::Settings, mb::Vector{Int64},_C1::Vector{Document},_C2::Vector{Document},
	_terms1::Vector{Vector{Int64}},_counts1::Vector{Vector{Int64}},_terms2::Vector{Vector{Int64}},_counts2::Vector{Vector{Int64}})


	init_γs!(model, mb)
	init_sstats!(model, settings)
	# @btime γ = copy(model._γ[mb])
	γ = model._γ[mb]
	elogΘ = dir_expectation2D.(γ)
	expelogΘ = expdot(elogΘ)
	elogϕ1 = deepcopy(model._elogϕ1)
	elogϕ2 = deepcopy(model._elogϕ2)
	expelogϕ1 = exp.(elogϕ1)
	expelogϕ2 = exp.(elogϕ2)
	change = 0.0
	πtemp = zeros(Float64, (model._K1, model._K2))
	sumπ1_d = zeros(Float64, (model._K1, model._K2))
	sumπ2_d = zeros(Float64, (model._K1, model._K2))
	sstat = zeros(Float64, (model._K1, model._K2))
	sstat_v1 = zeros(Float64, model._K1)
	sstat_v2 = zeros(Float64, model._K2)
	copyto!(model._sumπ1_mb, settings._zeroer_K1V1)
	copyto!(model._sumπ2_mb, settings._zeroer_K2V2)
	copyto!(πtemp, settings._zeroer_K1K2)
	copyto!(sumπ1_d, settings._zeroer_K1K2)
	copyto!(sumπ2_d, settings._zeroer_K1K2)
	copyto!(sstat, settings._zeroer_K1K2)
	copyto!(sstat_v1, zeros(Float64, model._K1))
	copyto!(sstat_v2, zeros(Float64, model._K2))
	for d in 1:length(mb)

		i = mb[d]
		doc1 = _C1[i]
		doc2 = _C2[i]
		ids1 = _terms1[i]
		cts1 = _counts1[i]
		ids2 = _terms2[i]
		cts2 = _counts2[i]
		γd = γ[d]
		elogΘd = elogΘ[d]
		expelogΘd = expelogΘ[d]

		for _ in 1:settings._MAX_GAMMA_ITER
			γ_old = deepcopy(γd)
			copyto!(sumπ1_d,settings._zeroer_K1K2)
			copyto!(sstat,settings._zeroer_K1K2)

			for (w,v) in enumerate(ids1)
				optimize_π_iw!(πtemp, expelogΘd, expelogϕ1[:,v])
				sstat = cts1[w] .* πtemp
				@. (sumπ1_d += sstat)
			end


			copyto!(sumπ2_d,settings._zeroer_K1K2)
			copyto!(sstat,settings._zeroer_K1K2)
			for (w,v) in enumerate(ids2)
				optimize_π_iw!(πtemp, expelogΘd, collect(expelogϕ2[:,v]'))
				sstat = cts2[w] .* πtemp
				@. (sumπ2_d += sstat)
			end


			optimize_γd!(model, γd,sumπ1_d,sumπ2_d)
			elogΘd .= dir_expectation2D(γd)
			expelogΘd .= exp.(elogΘd)
			change = mean_change(γd, γ_old)
			if change < settings._GAMMA_THRESHOLD
				break
			end
		end
		#d has converged
		#check whether this is necessary
		γ[d] .= γd
		elogΘ[d] .= elogΘd
		expelogΘ[d] .= expelogΘd

		# @btime sstat_dw = [(expelogΘd .* expelogϕ1[:,v]) .+ 1e-100 for v in ids1]
		# sstat_dw ./= sum.(sstat_dw)

		copyto!(sumπ1_d,settings._zeroer_K1K2)
		copyto!(sstat,settings._zeroer_K1K2)
		copyto!(sstat_v1, zeros(Float64, model._K1))
		for (w,v) in enumerate(ids1)

			optimize_π_iw!(πtemp, expelogΘd, expelogϕ1[:,v])
			sstat = cts1[w] .* πtemp
			@. (sumπ1_d += sstat)
			sstat_v1 .= sum(sstat , dims = 2)[:,1]
			@.(model._sumπ1_mb[:,v] += sstat_v1)
		end


		copyto!(sumπ2_d,settings._zeroer_K1K2)
		copyto!(sstat,settings._zeroer_K1K2)
		copyto!(sstat_v2, zeros(Float64, model._K1))

		for (w,v) in enumerate(ids2)
			optimize_π_iw!(πtemp, expelogΘd, collect(expelogϕ2[:,v]'))
			sstat = cts2[w] .* πtemp
			@. (sumπ2_d += sstat)
			sstat_v2 .= sum(sstat , dims = 1)[1,:]
			@.(model._sumπ2_mb[:,v] += sstat_v2)
		end


		optimize_γd!(model, γd,sumπ1_d,sumπ2_d)
		elogΘd .= dir_expectation2D(γd)
		expelogΘd .= exp.(elogΘd)
		γ[d] .= γd
		elogΘ[d] .= elogΘd
		expelogΘ[d] .= expelogΘd

		# sstat_dw = [optimize_π_iw!(πtemp,expelogΘd,expelogϕ1[:,v]) for v in ids1]
		# sstat_w = hcat(sum.(sstat_dw .* cts1, dims = 2)...)
		# model._sumπ1_mb[:,ids1] .+= sstat_w
		#
		# sstat_dw = [optimize_π_iw!(πtemp,expelogΘd,collect(expelogϕ2[:,v]')) for v in ids2]
		# sstat_w = collect(vcat(sum.(sstat_dw .* cts2, dims = 1)...)')
		# model._sumπ2_mb[:,ids2] .+= sstat_w
	end
	#check if needed
	copyto!(model._γ[mb],γ)
end

function update_global!(model::MVD, ρ::Float64, _D1::Int64, _D2::Int64, mb,len_mb2)
	copyto!(model._λ1_old,  model._λ1)
	optimize_λ!(model._λ1, length(mb), model._η1, model._sumπ1_mb, _D1)
	model._λ1 .= (1.0-ρ).*model._λ1_old .+ ρ.*model._λ1
	dir_expectationByRow!(model._elogϕ1,model._λ1)
	copyto!(model._λ2_old,model._λ2)
	optimize_λ!(model._λ2,len_mb2, model._η2, model._sumπ2_mb,_D2)
	model._λ2 .= (1.0-ρ).*model._λ2_old .+ ρ.*model._λ2
	dir_expectationByRow!(model._elogϕ2,model._λ2)
end
function calc_Θ_bar_d(obs1_dict::Dict{Int64,Array{Int64,1}}, obs2_dict::Dict{Int64,Array{Int64,1}},
	 d::Int64, model::MVD, count_params::TrainCounts,settings::Settings)
	doc1 = obs1_dict[d]
	doc2 = obs2_dict[d]
	corp1 = model._corpus1._docs[d]
	corp2 = model._corpus2._docs[d]
	γd = model._γ[d]
	γd .= 1.0
	init_sstats!(model, settings)
	elogϕ1 = deepcopy(model._elogϕ1)
	elogϕ2 = deepcopy(model._elogϕ2)
	expelogϕ1 = exp.(elogϕ1)
	expelogϕ2 = exp.(elogϕ2)
	elogΘd = dir_expectation2D(γd)
	expelogΘd = exp.(elogΘd)
	change = 0.0
	πtemp = deepcopy(settings._zeroer_K1K2)
	#############
	for _ in 1:settings._MAX_GAMMA_ITER
		γ_old = copy(γd)
		sumπ1_d = deepcopy(settings._zeroer_K1K2)
		obs_words_corp1inds = [find_all(d,corp1._terms)[1] for d in doc1]
		for (w,v) in enumerate(corp1._terms[obs_words_corp1inds])
			optimize_π_iw!(πtemp, expelogΘd, expelogϕ1[:,v])
			sstat = corp1._counts[w] .* πtemp
			@. (sumπ1_d += sstat)
		end
		sumπ2_d = deepcopy(settings._zeroer_K1K2)
		obs_words_corp2inds = [find_all(d,corp2._terms)[1] for d in doc2]
		for (w,v) in enumerate(corp2._terms[obs_words_corp2inds])
			optimize_π_iw!(πtemp, expelogΘd, collect(expelogϕ2[:,v]'))
			sstat = corp2._counts[w] .* πtemp
			@. (sumπ2_d += sstat)
		end
		optimize_γd_perp!(model, γd,sumπ1_d,sumπ2_d)
		elogΘd .= dir_expectation2D(γd)
		expelogΘd .= exp.(elogΘd)
		change = mean_change(γd, γ_old)
		if change < settings._GAMMA_THRESHOLD
			break
		end
	end

	copyto!(model._γ[d],γd)
	theta_bar = model._γ[d][:,:] ./ sum(model._γ[d])
	return theta_bar
end

function calc_perp(model::MVD,hos1_dict::Dict{Int64,Array{Int64,1}},
				   obs1_dict::Dict{Int64,Array{Int64,1}},hos2_dict::Dict{Int64,Array{Int64,1}},
				   obs2_dict::Dict{Int64,Array{Int64,1}},count_params::TrainCounts,
				   ϕ1_est::Matrix{Float64}, ϕ2_est::Matrix{Float64},settings::Settings)
	corp1 = deepcopy(model._corpus1)
	corp2 = deepcopy(model._corpus2)
	l1 = 0.0
	l2 = 0.0
	validation = collect(keys(hos1_dict))

	for d in validation


		theta_bar = calc_Θ_bar_d(obs1_dict, obs2_dict,d, model, count_params,settings)
		for v in hos1_dict[d]
			tmp = 0.0
			for k in 1:count_params._K1
				tmp += ((ϕ1_est[k,v]*sum(theta_bar, dims=2)[k,1]))
			end
			l1 += log(tmp)
		end
		for v in hos2_dict[d]
			tmp = 0.0
			for k in 1:count_params._K2
				tmp += ((ϕ2_est[k,v]*sum(theta_bar, dims=1)[1,k]))
			end
			l2 += log(tmp)
		end

	end
	l1/= sum(length.(collect(values(hos1_dict))))
	l2/= sum(length.(collect(values(hos2_dict))))

	return exp(-l1), exp(-l2)
end

function update_α_newton_iterative!(model::MVD, count_params::TrainCounts, h_map::Vector{Bool}, settings::Settings)
	D = count_params._D1
	@assert D == sum(.!h_map)
	logphat = vectorize_mat(sum(dir_expectation2D.(model._γ[.!h_map]))./D)
	counter = 0
	cond = true
	decay = 0
	K = prod(size(model._α))
	Alpha = ones(Float64, K)
	# Alpha = vectorize_mat(mean(model._γ[.!h_map])./25.0)
	Alpha_new = zeros(Float64,K)
	ga = zeros(Float64,K)
	Ha = zeros(Float64,K)

	while cond
		sumgh = 0.0
		sum1h = 0.0
		for k in 1:K
			# global sumgh, sum1h
			ga[k] = D*(Ψ(sum(Alpha))- Ψ(Alpha[k]) + logphat[k])
			Ha[k] = -D*dΨ(Alpha[k])
			sumgh += ga[k]/Ha[k]
			sum1h += 1.0/Ha[k]
		end
		z = D*dΨ(sum(Alpha))
		c = sumgh/(1.0/z + sum1h)
		while true
			singular = false
			for k in 1:K
				# global singular
				step_ = (settings._ALPHA_DECAY_FACTOR^decay) * (ga[k] - c)/Ha[k]
				if Alpha[k] <= step_
					singular = true
					break
				end
				Alpha_new[k] = Alpha[k] - step_
			end
			singular
			if singular
				decay += 1
				copyto!(Alpha_new,Alpha)
				if decay > settings._MAX_ALPHA_DECAY
					break
				end
			else
				break;
			end
		end
		cond = false
		mchange_ = mean_change(Alpha_new, Alpha)
		if mchange_ >= settings._ALPHA_THRESHOLD
			cond =true
		end
		if counter > settings._MAX_ALPHA_ITER
			cond = false
		end
		if decay > settings._MAX_ALPHA_DECAY
			break;
		end
		counter += 1
		copyto!(Alpha, Alpha_new)
	end
	copyto!(model._α, matricize_vec(Alpha, model._K1, model._K2))

end

function update_α_newton_mb!(model::MVD,ρ, _D1::Int64,mb::Vector{Int64}, h_map::Vector{Bool}, settings::Settings)
	D = _D1
	n = length(mb)
	logphat = vectorize_mat(sum(dir_expectation2D.(model._γ[mb]))./n)

	K = prod(size(model._α))
	Alpha = vectorize_mat(deepcopy(model._α))
	Alpha_new = zeros(Float64,K)
	ga = zeros(Float64,K)
	Ha = zeros(Float64,K)
	# sumgh = 0.0
	# sum1h = 0.0
	#
	# for k in 1:K
	# 	global sumgh, sum1h
	# 	ga[k] = D*(Ψ(sum(Alpha))- Ψ(Alpha[k]) + logphat[k])
	# 	Ha[k] = -D*dΨ(Alpha[k])
	# 	sumgh += ga[k]/Ha[k]
	# 	sum1h += 1.0/Ha[k]
	# end
	ga .= D .* (Ψ(sum(Alpha)) .- Ψ.(Alpha) .+ logphat)
	Ha .= -D .*dΨ.(Alpha)
	sumgh = sum(ga./Ha)
	sum1h = sum(1.0./Ha)
	z = D*dΨ(sum(Alpha))
	c = sumgh/(1.0/z + sum1h)
	step_ = ρ .* (ga .- c)./Ha
	if all(Alpha .> step_)
		Alpha_new = Alpha .- step_
		copyto!(model._α, matricize_vec(Alpha_new, model._K1, model._K2))
	end
end

function update_η1_newton_mb!(model::MVD,ρ, settings::Settings)
	D = size(model._η1,1)
	logphat = (sum(dir_expectation(model._λ1[k,:]) for k in 1:D) ./ D)
	V = model._corpus1._V
	#Alpha = ones(Float64, 25)
	eta1 = deepcopy(model._η1[1,:])
	eta1_new = zeros(Float64,V)
	ga = zeros(Float64,V)
	Ha = zeros(Float64,V)


	sumgh = 0.0
	sum1h = 0.0
	for v in 1:V
		#global sumgh, sum1h
		ga[v] = D*(Ψ(sum(eta1))- Ψ(eta1[v]) + logphat[v])
		Ha[v] = -D*dΨ(eta1[v])
		sumgh += ga[v]/Ha[v]
		sum1h += 1.0/Ha[v]
	end
	z = D*dΨ(sum(eta1))
	c = sumgh/(1.0/z + sum1h)

	step_ = ρ .* (ga .- c)./Ha
	if all(eta1 .> step_)
		eta1_new = eta1 .- step_
		copyto!(model._η1, collect(repeat(eta1_new, inner=(1,D))'))
	end
end
function update_η2_newton_mb!(model::MVD,ρ,  settings::Settings)
	D = size(model._η2,1)
	logphat = (sum(dir_expectation(model._λ2[k,:]) for k in 1:D) ./ D)
	V = model._corpus2._V
	eta2 = deepcopy(model._η2[1,:])
	eta2_new = zeros(Float64,V)
	ga = zeros(Float64,V)
	Ha = zeros(Float64,V)


	sumgh = 0.0
	sum1h = 0.0
	for v in 1:V
		#global sumgh, sum1h
		ga[v] = D*(Ψ(sum(eta2))- Ψ(eta2[v]) + logphat[v])
		Ha[v] = -D*dΨ(eta2[v])
		sumgh += ga[v]/Ha[v]
		sum1h += 1.0/Ha[v]
	end
	z = D*Ψ(sum(eta2))
	c = sumgh/(1.0/z + sum1h)

	step_ = ρ .* (ga .- c)./Ha
	if all(eta2 .> step_)
		eta2_new = eta2 .- step_
		copyto!(model._η2, collect(repeat(eta2_new, inner=(1,D))'))
	end
end
