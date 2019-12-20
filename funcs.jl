function init_γs!(model::MVD, mb::Vector{Int64})
	for i in mb
		# model.γ[i] .= 1.0
		model._γ[i] = rand(Gamma(100.0, 0.01), (model._K1,model._K2))
	end
end

function init_sstats!(model::MVD, settings::Settings, mb::Vector{Int64})
	copyto!(model._sumπ1_mb, settings._zeroer_K1V1)
	copyto!(model._sumπ2_mb, settings._zeroer_K2V2)
	# πtemp = zeros(Float64, (model._K1, model._K2));
	sumπ1_d = zeros(Float64, model._K1* model._K2);
	sumπ2_d = zeros(Float64, model._K1* model._K2);
	γ = deepcopy(model._γ[mb])
	γd = zeros(Float64, model._K1* model._K2)
	γ_old = zeros(Float64, model._K1*model._K2)
	elogΘd =zeros(Float64, model._K1*model._K2);
	expelogΘd = zeros(Float64, model._K1* model._K2);
	return sumπ1_d,sumπ2_d,γ,γd,γ_old,elogΘd,expelogΘd
end



function update_local!(model::MVD,settings::Settings, mb::Vector{Int64},_terms1::Vector{Vector{Int64}},
	_counts1::Vector{Vector{Int64}},_terms2::Vector{Vector{Int64}},_counts2::Vector{Vector{Int64}},args...)

	change = 0.0
	sumπ1_d,sumπ2_d,γ,γd,γ_old,elogΘd,expelogΘd = args
	expelogϕ1 = exp.(model._elogϕ1)
	expelogϕ2 = exp.(model._elogϕ2)
	expelogϕ1d = repeat(expelogϕ1,inner = (model._K2,1))
	expelogϕ2d = repeat(expelogϕ2,inner = (model._K1,1))
	cs = get1DColIndices(model._K1, model._K2)
	rs = get1DRowIndices(model._K1, model._K2)
	for d in 1:length(mb)
		i = mb[d]
		ids1, cts1 = _terms1[i], _counts1[i]
		ids2, cts2 = _terms2[i], _counts2[i]
		γd = vectorize_mat(γ[d])
		γd2 = vectorize_mat(γ[d]')
		elogΘd1 = dir_𝔼log_shifted(γd)
		elogΘd2 = dir_𝔼log_shifted(γd2)
		expelogΘd1 = exp.(elogΘd1)
		expelogΘd2 = exp.(elogΘd2)
		phinorm1 = collect(expelogΘd1' * expelogϕ1d[:,ids1])[1,:] .+ 1e-100
		phinorm2 = collect(expelogΘd2' * expelogϕ2d[:,ids2])[1,:] .+ 1e-100
		if sum(cts2) != 0
			settings._MAX_GAMMA_ITER = 1000
			settings._GAMMA_THRESHOLD = 1e-3
		else
			settings._MAX_GAMMA_ITER = 100000
			settings._GAMMA_THRESHOLD = 1e-10
		end

		for _ in 1:settings._MAX_GAMMA_ITER
			# global sumπ1_d,sumπ2_d,γ_old,γd, γd2,elogΘd1,expelogΘd1,elogΘd2,expelogΘd2,phinorm1,phinorm2,change
			γ_old = deepcopy(γd)
			sumπ1_d = (expelogΘd1 .* (collect((cts1./phinorm1)' * expelogϕ1d[:,ids1]')[1,:]))[rs]
			if sum(cts2) != 0
				sumπ2_d = (expelogΘd2 .* (collect((cts2./phinorm2)' * expelogϕ2d[:,ids2]')[1,:]) )[cs]
			else
				sumπ2_d = ones(Float64, (model._K1 , model._K2)) .* 1e-100
			end
			mat = model._α  .+ sumπ1_d .+ sumπ2_d
			γd = vectorize_mat(mat)
			γd2 = vectorize_mat(mat')
			elogΘd1 = dir_𝔼log_shifted(γd)
			expelogΘd1 = exp.(elogΘd1)
			elogΘd2 = dir_𝔼log_shifted(γd2)
			expelogΘd2 = exp.(elogΘd2)
			phinorm1 = collect(expelogΘd1' * expelogϕ1d[:,ids1])[1,:] .+ 1e-100
			phinorm2 = collect(expelogΘd2' * expelogϕ2d[:,ids2])[1,:] .+ 1e-100
			change = mean_change(γd, γ_old)
			# println(change)
			if change < settings._GAMMA_THRESHOLD
				break
			end
		end
		#d has converged
		copyto!(model._γ[i],matricize_vec(γd, model._K1, model._K2))
		s = expelogΘd1 * (cts1./phinorm1)'
		for (j,id) in enumerate(ids1)
			model._sumπ1_mb[:,id] .+= sum(s[rs,j], dims = 2)[:,1]
		end
		if sum(cts2) != 0
			s = expelogΘd2 * (cts2./phinorm2)'

			for (j,id) in enumerate(ids2)
				model._sumπ2_mb[:,id] .+= sum(s[rs,j], dims = 2)[:,1]
			end
		else
			model._sumπ2_mb .+= ones(Float64, size(model._sumπ2_mb)) .* 1e-100   ##Added
		end
	end
	model._sumπ1_mb .*= expelogϕ1
	model._sumπ1_mb .+= 1e-100
	model._sumπ2_mb .*= expelogϕ2
	model._sumπ2_mb .+= 1e-100
	print("")
end

function optimize_λ!(lambda_::Matrix{Float64},_d::Int64, model_eta::Matrix{Float64}, sum_π_mb::Matrix{Float64},_D::Int64)
	copyto!(lambda_, model_eta)
	@.(lambda_ +=  (_D/_d) * sum_π_mb)
end

function update_global!(model::MVD, ρ::Float64, _D1::Int64, _D2::Int64, _d1::Int64,_d2::Int64)
	copyto!(model._λ1_old,  model._λ1)
	optimize_λ!(model._λ1, _d1, model._η1, model._sumπ1_mb, _D1)
	model._λ1 .= (1.0-ρ).*model._λ1_old .+ ρ.*model._λ1
	dir_𝔼log_row_shifted!(model._elogϕ1,model._λ1)    # <=====  HERE
	copyto!(model._λ2_old,model._λ2)
	optimize_λ!(model._λ2,_d2, model._η2, model._sumπ2_mb,_D2)
	model._λ2 .= (1.0-ρ).*model._λ2_old .+ ρ.*model._λ2
	dir_𝔼log_row_shifted!(model._elogϕ2,model._λ2)    # <=====  HERE
end

function get_holdout_Θd(obs1_dict::Dict{Int64,Array{Int64,1}}, obs2_dict::Dict{Int64,Array{Int64,1}},
	 d::Int64, model::MVD, count_params::TrainCounts,settings::Settings, _terms1, _terms2, _counts1, _counts2)

	cs = get1DColIndices(model._K1, model._K2)
 	rs = get1DRowIndices(model._K1, model._K2)
	args = init_sstats!(model, settings, [d])
	sumπ1_d,sumπ2_d,γ,γd,γ_old,elogΘd,expelogΘd = args
	doc1 = obs1_dict[d]
	doc2 = obs2_dict[d]
	corp1 = model._corpus1._docs[d]
	corp2 = model._corpus2._docs[d]
	obs_words_corp1inds = [find_all(d,corp1._terms)[1] for d in doc1]
	obs_words_corp2inds = [find_all(d,corp2._terms)[1] for d in doc2]
	ids1, cts1 = _terms1[d][obs_words_corp1inds], _counts1[d][obs_words_corp1inds]
	ids2, cts2 = _terms2[d][obs_words_corp2inds], _counts2[d][obs_words_corp2inds]

	γd = vectorize_mat(model._γ[d]); elogΘd1 = dir_𝔼log_shifted(γd); expelogΘd1 = exp.(elogΘd1)
	γd2 = vectorize_mat(model._γ[d]') ; elogΘd2 = dir_𝔼log_shifted(γd2); expelogΘd2 = exp.(elogΘd2)
	expelogϕ1 = exp.(model._elogϕ1) ; expelogϕ1d = repeat(expelogϕ1,inner = (model._K2,1))
	expelogϕ2 = exp.(model._elogϕ2) ; expelogϕ2d = repeat(expelogϕ2,inner = (model._K1,1))

	change = 0.0


	phinorm1 = collect(expelogΘd1' * expelogϕ1d[:,ids1])[1,:] .+ 1e-100
	phinorm2 = collect(expelogΘd2' * expelogϕ2d[:,ids2])[1,:] .+ 1e-100

	if sum(cts2) != 0
		settings._MAX_GAMMA_ITER = 1000
		settings._GAMMA_THRESHOLD = 1e-3
	else
		settings._MAX_GAMMA_ITER = 100000
		settings._GAMMA_THRESHOLD = 1e-10
	end
	#############
	for _ in 1:settings._MAX_GAMMA_ITER
		γ_old = deepcopy(γd)
		γ_old = deepcopy(γd)
		sumπ1_d = (expelogΘd1 .* (collect((cts1./phinorm1)' * expelogϕ1d[:,ids1]')[1,:]))[rs]
		if sum(cts2) != 0
			sumπ2_d = (expelogΘd2 .* (collect((cts2./phinorm2)' * expelogϕ2d[:,ids2]')[1,:]) )[cs]
		else
			sumπ2_d = zeros(Float64, (model._K1 , model._K2)) .* 1e-100
		end
		mat = model._α  .+ sumπ1_d .+ sumπ2_d
		γd = vectorize_mat(mat)
		γd2 = vectorize_mat(mat')
		elogΘd1 = dir_𝔼log_shifted(γd)
		expelogΘd1 = exp.(elogΘd1)
		elogΘd2 = dir_𝔼log_shifted(γd2)
		expelogΘd2 = exp.(elogΘd2)
		phinorm1 = collect(expelogΘd1' * expelogϕ1d[:,ids1])[1,:] .+ 1e-100
		phinorm2 = collect(expelogΘd2' * expelogϕ2d[:,ids2])[1,:] .+ 1e-100
		change = mean_change(γd, γ_old)
		if change < settings._GAMMA_THRESHOLD
			break
		end
	end
	#d has converged
	copyto!(model._γ[d],matricize_vec(γd, model._K1, model._K2))

	theta_bar = model._γ[d][:,:] ./ sum(model._γ[d])
	return theta_bar
end
function calc_perp(model::MVD,hos1_dict::Dict{Int64,Array{Int64,1}},
				   obs1_dict::Dict{Int64,Array{Int64,1}},hos2_dict::Dict{Int64,Array{Int64,1}},
				   obs2_dict::Dict{Int64,Array{Int64,1}},count_params::TrainCounts,
				   ϕ1_est::Matrix{Float64}, ϕ2_est::Matrix{Float64},settings::Settings, _terms1, _terms2, _counts1, _counts2)
	corp1 = deepcopy(model._corpus1)
	corp2 = deepcopy(model._corpus2)
	l1 = 0.0
	l2 = 0.0
	validation = collect(keys(hos1_dict))

	for d in validation
		theta_bar = get_holdout_Θd(obs1_dict, obs2_dict,d, model, count_params,settings, _terms1, _terms2, _counts1, _counts2)
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
function evaluate_at_epoch(folder,model, h_map, settings, count_params,hos1_dict,obs1_dict,hos2_dict,obs2_dict,perp1_list,perp2_list,VI_CONVERGED, epoch_count,
 	_terms1, _terms2, _counts1, _counts2)

	ϕ1_est = mean_dir_by_row(model._λ1)
	ϕ2_est = mean_dir_by_row(model._λ2)
	@info "computing perplexity..."
	p1, p2 = calc_perp(model,hos1_dict,obs1_dict,hos2_dict,obs2_dict,
	count_params, ϕ1_est, ϕ2_est, settings, _terms1, _terms2, _counts1, _counts2)
	perp1_list = vcat(perp1_list, p1)
	@info "perp1 = $(p1)"
	perp2_list = vcat(perp2_list, p2)
	@info "perp2 = $(p2)"
	@save "$(folder)/perp1_at_$(epoch_count)"  perp1_list
	@save "$(folder)/perp2_at_$(epoch_count)"  perp2_list
	@save "$(folder)/model_at_epoch_$(epoch_count)"  model
	if length(perp1_list) > 2
		if (abs(perp1_list[end]-perp1_list[end-1])/perp1_list[end] < settings._VI_THRESHOLD) &&
			(abs(perp2_list[end]-perp2_list[end-1])/perp2_list[end] < settings._VI_THRESHOLD)
			VI_CONVERGED  = true
		end
	end
	return VI_CONVERGED
end


function update_α_newton_full!(model::MVD, complete_train_ids::Vector{Int64}, settings::Settings)
	@info "updating α"
	mba = complete_train_ids
	D = length(mba);K = prod(size(model._α));
	α_new = zeros(Float64,(100,K));
	norm_old = 100.0
	for j in 1:100
		# println(j)
		α = ones(Float64, K) .* (1.0/K)
		ν = K
		for _ in 1:50
			# global norm_old
			ρ = .8
			∇α = [(ν+.5)/ (.5+α[k]) + D .* (Ψ(sum(α .+ .5))-Ψ(α[k] + .5)) for k in 1:K] + vectorize_mat(sum(dir_𝔼log_shifted.(model._γ[mba])))
			Hα_inv_diag = -1.0 ./ (D * dΨ.(α .+ .5) + (ν+.5) ./ (α .+ .5).^2)
			p = (∇α .- dot(∇α, Hα_inv_diag) / (1.0 / (D * dΨ(sum(α)+.5)) + sum(Hα_inv_diag))) .* Hα_inv_diag
			# ρ *= ρ
			# minimum(α - ρ * p .- .5)
			while minimum(α - ρ * p .- .5) <= -0.5
				# global ρ
				ρ *= ρ
			end
			α .-= (ρ * p)
			if (abs((norm(∇α)/D) - norm_old) < settings._ALPHA_THRESHOLD) && (ν < settings._ALPHA_THRESHOLD)
				α_new[j,:] .= deepcopy(α)
				break;
			end
			ν *= .5
			norm_old = norm(∇α)/D
			α_new[j,:] .= deepcopy(α)
		end
	end

	# display(matricize_vec(median(skipmissing(α_new).x, dims = 1)[1,:], 5, 5)[inds1, inds2])
	model._α .= matricize_vec(median(skipmissing(α_new).x, dims = 1)[1,:], model._K1, model._K2)
end
function update_α_newton_full_one_shot!(model::MVD, complete_train_ids::Vector{Int64}, settings::Settings)
	@info "updating α"
	mba = complete_train_ids
	D = length(mba);K = prod(size(model._α));
	norm_old = 100.0
	α = vectorize_mat(mean(model._γ[mba]))./K;
	α./=sum(α)##ADDED


	α[α .< 1/K] .= 1e-5
	ν = K
	for _ in 1:50
		ρ = .8
		∇α = [(ν+.5)/ (.5+α[k]) + D .* (Ψ(sum(α .+ .5))-Ψ(α[k] + .5)) for k in 1:K] + vectorize_mat(sum(dir_𝔼log_shifted.(model._γ[mba])))
		Hα_inv_diag = -1.0 ./ (D * dΨ.(α .+ .5) + (ν+.5) ./ (α .+ .5).^2)
		p = (∇α .- dot(∇α, Hα_inv_diag) / (1.0 / (D * dΨ(sum(α)+.5)) + sum(Hα_inv_diag))) .* Hα_inv_diag
		# minimum(α - ρ * p .- .5)
		# ρ *= ρ
		while minimum(α - ρ * p .- .5) <= -0.5
			ρ *= ρ
			if ρ == 0.0
				break;
			end
		end
		α .-= (ρ * p)
		if (abs((norm(∇α)/D) - norm_old) < settings._ALPHA_THRESHOLD) && (ν < settings._ALPHA_THRESHOLD)
			break;
		end
		ν *= .5
		norm_old = norm(∇α)/D
	end
	α./=sum(α)###ADDED
	# display(matricize_vec(median(skipmissing(α).x, dims = 1)[1,:], 5, 5)[inds1, inds2])
	model._α .= matricize_vec(α, model._K1, model._K2)
end




# function update_α_newton_mb!(model::MVD,ρ, _D1::Int64,mb::Vector{Int64}, h_map::Vector{Bool}, settings::Settings)
#
# 	D = _D1 ;n = length(mb);K = prod(size(model._α));
# 	# α = vectorize_mat(deepcopy(model._α));
# 	α = rand(Gamma(100.0, 0.01), model._K1*model._K2)
# 	α_new = zeros(Float64,K);
# 	α_shifted = α .+ .5
#
	# sstats = vectorize_mat(mean(dir_𝔼log_shifted.(model._γ[mb])))
#
# 	# g = D .* (sstats .- dir_𝔼log_shifted(α .+ .5))
# 	# g = D .* (sstats .- dir_𝔼log(α_shifted))
# 	g = (sstats .- dir_𝔼log(α_shifted))
# 	# H = -D .*dΨ.(α .+ .5)
# 	# H = -D .*dΨ.(α_shifted)
# 	H .= dΨ.(sum(α_shifted))
# 	H = -dΨ.(α_shifted)
#
#
# 	_gH = sum(g./H); _1H = sum(1.0./H);
# 	# z = D*dΨ(sum(α)) ; c = _gH/(1.0/z + _1H)
# 	# z = D*dΨ(sum(α_shifted)) ; c = _gH/(1.0/z + _1H)
# 	z = dΨ(sum(α_shifted)) ; c = _gH/(1.0/z + _1H)
#
# 	# step_ = ρ .* (g .- c)./H
# 	step_ = (g .- c)./H
# 	x = 0
# 	while any(α_shifted .- (step_./(10^x)) .< 0.5)
# 		global x;
# 		x+=1
# 	end
# 	println(x)
# 	println(norm(g))
# 		α_shifted .-= (step_./(10^x));
# 		α_new = α_shifted .- .5;
# 		copyto!(model._α, matricize_vec(α_new, model._K1, model._K2))
# 		print("tell me")
# 	end
# 	println(norm(g))
# end


# function update_α_newton_mb2!(model::MVD,ρ, _D1::Int64,mb::Vector{Int64}, h_map::Vector{Bool}, settings::Settings)
# 	K = prod(size(model._α))
# 	iq = zeros(Float64, K)
# 	g = zeros(Float64, K)
#
# 	α = vectorize_mat(model._α)
# 	#α[α .== 1e-20] .= 1e-12
# 	α0 = sum(α)
# 	# elogp =  vectorize_mat(mean(dir_expectation2D.(model._γ[mb])))
# 	elogp =  vectorize_mat(mean(dir_𝔼log.(model._γ[mb])))
# 	converged = false
# 	while !converged
#
# 		iz = 1.0/dΨ(α0)
# 		gnorm = 0.0
# 		b = 0.0
# 		iqs = 0.0
# 		for k in 1:K
# 			global b, iqs, gnorm, elogp, iz
# 			ak = α[k]
# 			g[k] = gk = -dir_𝔼log_shifted(α)[k] + elogp[k]
# 			iq[k] = -1.0/dΨ(ak)
# 			b += gk*iq[k]
# 			iqs += iq[k]
# 			agk = abs(gk)
# 			if agk > gnorm
# 				gnorm = agk
# 			end
# 		end
# 		b /= (iz + iqs)
# 		for k in 1:K
# 			α[k] -= (g[k] - b)*iq[k]
# 			if α[k] < 1e-12
# 				α[k] = 1e-12
# 			end
# 		end
# 		α0 = sum(α)
# 		converged = gnorm < 1e-5
# 		gnorm
# 	end
# 	return α
# end


# function update_η1_newton_mb!(model::MVD,ρ, settings::Settings)
# 	D = size(model._η1,1); V = model._corpus1._V
# 	η1 = deepcopy(model._η1[1,:]) ;η1_new = zeros(Float64,V);
#
# 	sstats = (sum(dir_𝔼log(model._λ1[k,:]) for k in 1:D) ./ D)    # <=====  HERE
#
# 	g = D .* (sstats .- dir_𝔼log_shifted(η1))
# 	H = -D .* dΨ.(η1)
# 	_gH = sum(g ./ H) ; _1H = sum(1.0./H)
# 	z = D*dΨ(sum(η1))
# 	c = _gH/(1.0/z + _1H)
#
# 	step_ = ρ .* (g .- c)./H
#
# 	if all(η1 .> step_)
# 		η1_new = η1 .- step_
# 		copyto!(model._η1, collect(repeat(η1_new, inner=(1,D))'))
# 	end
# end

# function update_η2_newton_mb!(model::MVD,ρ,  settings::Settings)
# 	D = size(model._η2,1); V = model._corpus2._V
# 	η2 = deepcopy(model._η2[1,:]) ;η2_new = zeros(Float64,V);
#
# 	sstats = (sum(dir_𝔼log(model._λ2[k,:]) for k in 1:D) ./ D)    # <=====  HERE
#
# 	g = D .* (sstats .- dir_𝔼log_shifted(η2))
# 	H = -D .* dΨ.(η2)
# 	_gH = sum(g ./ H) ; _1H = sum(1.0./H)
# 	z = D*dΨ(sum(η2))
# 	c = _gH/(1.0/z + _1H)
#
# 	step_ = ρ .* (g .- c)./H
#
# 	if all(η2 .> step_)
# 		η2_new = η2 .- step_
# 		copyto!(model._η2, collect(repeat(η2_new, inner=(1,D))'))
# 	end
# end


print("")

function pϕ_𝔼log(model, _train_ids)
	D = length(_train_ids)
	elogphi1  = zero(model._elogϕ1)
	dir_𝔼log_row!(elogphi1, model._λ1)
	elogphi2  = zero(model._elogϕ2)
	dir_𝔼log_row!(elogphi2, model._λ2)
	(-sum([Dirichlet(model._η1[k,:]).lmnB for k in 1:model._K1]) + sum((model._η1 .- 1.0) .* elogphi1)
	-sum([Dirichlet(model._η2[k,:]).lmnB for k in 1:model._K2]) + sum((model._η2 .- 1.0) .* elogphi2))/D

end
function qϕ_𝔼log(model, _train_ids)
	D = length(_train_ids)
	# elogphi1  = zero(model._elogϕ1)
	# dir_𝔼log_row!(elogphi1, model._λ1)
	(sum([-entropy(Dirichlet(model._λ1[k,:])) for k in 1:model._K1]) +
	sum([-entropy(Dirichlet(model._λ2[k,:])) for k in 1:model._K2]))/D
end
function pΘ_𝔼log(model, _train_ids)
	D = length(_train_ids)
	α = vectorize_mat(model._α)

	elogΘ = collect(hcat(dir_𝔼log.(vectorize_mat.(model._γ[_train_ids]))...))

	(-sum([Dirichlet(α).lmnB for _ in 1:D]) + sum((α .- 1.0) .* elogΘ)) / D
end
function qΘ_𝔼log(model, _train_ids)
	D = length(_train_ids)
	γs = collect(hcat(vectorize_mat.(model._γ[_train_ids])...))
	sum([-entropy(Dirichlet(γs[:,d])) for d in 1:D])/D

end

function get_elbo(model, _train_ids)
	pϕ_𝔼log(model, _train_ids) - qϕ_𝔼log(model, _train_ids) + pΘ_𝔼log(model, _train_ids) - qΘ_𝔼log(model, _train_ids)
end

# function pz_𝔼log()
# end
# function qz_𝔼log()
# end
# function pw_𝔼log()
# end
print("")
