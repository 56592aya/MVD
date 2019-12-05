"""
    init_γ!(model, mb)

initialize `γ` for a minibatch `mb`
"""
function init_γs!(model::MVD, mb::Vector{Int64})
	for i in mb
		# model.γ[i] .= 1.0
		model._γ[i] = rand(Gamma(100.0, 0.01), (model._K1,model._K2))
	end
end
# """
#     init_sstats!(model, settings)
#
# initialize statistics before the local update of variational parameters
# """
# function init_sstats!(model::MVD, settings::Settings, mb::Vector{Int64})
# 	copyto!(model._sumπ1_mb, settings._zeroer_K1V1)
# 	copyto!(model._sumπ2_mb, settings._zeroer_K2V2)
# 	πtemp = zeros(Float64, (model._K1, model._K2));
# 	sumπ1_d = zeros(Float64, (model._K1, model._K2));
# 	sumπ2_d = zeros(Float64, (model._K1, model._K2));
# 	sstat = zeros(Float64, (model._K1, model._K2));
# 	sstat_v1 = zeros(Float64, model._K1);
# 	sstat_v2 = zeros(Float64, model._K2);
# 	γ = deepcopy(model._γ[mb])
# 	γd = zeros(Float64, (model._K1, model._K2))
# 	γ_old = zeros(Float64, (model._K1, model._K2))
# 	elogΘd =zeros(Float64, (model._K1, model._K2));
# 	expelogΘd = zeros(Float64, (model._K1, model._K2));
# 	return πtemp,sumπ1_d,sumπ2_d,sstat,sstat_v1,sstat_v2,γ,γd,γ_old,elogΘd,expelogΘd
# end
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
"""
    optimize_γd!(model, γ,sumπ1_d,sumπ2_d)

optimize the variational parameter of γ for a document in the training
"""
function optimize_γd!(model, γd,sumπ1_d,sumπ2_d)
	@. (γd =  model._α + sumπ1_d + sumπ2_d)
end
"""
    optimize_γd_perp!(model, γ,sumπ1_d,sumπ2_d)

optimize the variational parameter of `γ` for a document in heldout
"""
function optimize_γd_perp!(model, γd,sumπ1_i,sumπ2_i)
	@. (γd =  model._α + sumπ1_i + sumπ2_i)
end
"""
    optimize_λ!(λ, mbsize,η,sum,D)

optimize the global variational parameter of `λ` given the current batch documents statistics `sum`
"""
function optimize_λ!(lambda_::Matrix{Float64},_d::Int64, model_eta::Matrix{Float64}, sum_π_mb::Matrix{Float64},_D::Int64)
	copyto!(lambda_, model_eta)
	@.(lambda_ +=  (_D/_d) * sum_π_mb)
end
"""
    optimize_π_iw!(π, expelogΘ,expelogϕ)

optimize the variational local parameter `π_dwk`
"""
function optimize_π_iw!(πtemp, expelogΘ, expelogϕ)
	@. (πtemp = dot(expelogΘ, expelogϕ) + 1e-100)
	# println(sum(πtemp))
	πtemp ./= sum(πtemp)
end
"""
    update_local!(model,settings,mb, terms1, counts1, terms2, counts2)

updating the local variational parameters `γ` and `π` for the given bacth `mb` \n
in the train stage
"""


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

		elogΘd1 = sum(cts2) != 0 ? dir_expectation_shifted(γd) : dir_expectation_shifted(γd) ##change second to unshifted
		expelogΘd1 = exp.(elogΘd1)
		elogΘd2 = sum(cts2) != 0 ? dir_expectation_shifted(γd2) : dir_expectation_shifted(γd2)
		expelogΘd2 = exp.(elogΘd2)
		phinorm1 = collect(expelogΘd1' * expelogϕ1d[:,ids1])[1,:] .+ 1e-100
		phinorm2 = collect(expelogΘd2' * expelogϕ2d[:,ids2])[1,:] .+ 1e-100

		for _ in 1:settings._MAX_GAMMA_ITER
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
			elogΘd1 = sum(cts2) != 0 ? dir_expectation_shifted(γd) : dir_expectation_shifted(γd)##change to unshifted
			expelogΘd1 = exp.(elogΘd1)
			elogΘd2 = sum(cts2) != 0 ? dir_expectation_shifted(γd2) : dir_expectation_shifted(γd2)##change to unshifted
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
"""
    update_global!(model,ρ,D1, D2, _d1, _d2)

noisy updating the global variational parameters λ after seeing bacth `mb` \n
in the train stage using a learning rate `ρ`
"""
function update_global!(model::MVD, ρ::Float64, _D1::Int64, _D2::Int64, _d1::Int64,_d2::Int64)
	copyto!(model._λ1_old,  model._λ1)
	optimize_λ!(model._λ1, _d1, model._η1, model._sumπ1_mb, _D1)
	model._λ1 .= (1.0-ρ).*model._λ1_old .+ ρ.*model._λ1
	dir_expectationByRow_shifted!(model._elogϕ1,model._λ1)    # <=====  HERE
	copyto!(model._λ2_old,model._λ2)
	optimize_λ!(model._λ2,_d2, model._η2, model._sumπ2_mb,_D2)
	model._λ2 .= (1.0-ρ).*model._λ2_old .+ ρ.*model._λ2
	dir_expectationByRow_shifted!(model._elogϕ2,model._λ2)    # <=====  HERE
end
"""
    get_holdout_Θd!(obs1,obs2,model, counts, d, settings)

updates the `γd` for the a document in the holdout using the observed terms in the
	test document `d` and returns mean `Θd`.\n
"""
function get_holdout_Θd(obs1_dict::Dict{Int64,Array{Int64,1}}, obs2_dict::Dict{Int64,Array{Int64,1}},
	 d::Int64, model::MVD, count_params::TrainCounts,settings::Settings)
	doc1 = obs1_dict[d]
	doc2 = obs2_dict[d]
	corp1 = model._corpus1._docs[d]
	corp2 = model._corpus2._docs[d]
	γd = deepcopy(model._γ[d])
	γd .= 1.0
	init_sstats!(model, settings, [d])
	elogϕ1 = deepcopy(model._elogϕ1)
	elogϕ2 = deepcopy(model._elogϕ2)
	expelogϕ1 = exp.(elogϕ1)
	expelogϕ2 = exp.(elogϕ2)
	elogΘd = dir_expectation2D(γd)
	expelogΘd = exp.(elogΘd)
	change = 0.0
	πtemp = zeros(Float64, (model._K1, model._K2));copyto!(πtemp, settings._zeroer_K1K2);
	sumπ1_d = zeros(Float64, (model._K1, model._K2));copyto!(sumπ1_d, settings._zeroer_K1K2);
	sumπ2_d = zeros(Float64, (model._K1, model._K2));copyto!(sumπ2_d, settings._zeroer_K1K2);
	sstat = zeros(Float64, (model._K1, model._K2));copyto!(sstat, settings._zeroer_K1K2);
	#############
	for _ in 1:settings._MAX_GAMMA_ITER
		γ_old = deepcopy(γd)
		copyto!(sumπ1_d,settings._zeroer_K1K2)
		copyto!(sstat,settings._zeroer_K1K2)
		obs_words_corp1inds = [find_all(d,corp1._terms)[1] for d in doc1]
		for (w,v) in enumerate(corp1._terms[obs_words_corp1inds])
			copyto!(πtemp, settings._zeroer_K1K2);
			optimize_π_iw!(πtemp, expelogΘd, expelogϕ1[:,v])
			sstat = corp1._counts[w] .* πtemp
			@. (sumπ1_d += sstat)
		end
		copyto!(sumπ2_d,settings._zeroer_K1K2)
		copyto!(sstat,settings._zeroer_K1K2)
		obs_words_corp2inds = [find_all(d,corp2._terms)[1] for d in doc2]
		for (w,v) in enumerate(corp2._terms[obs_words_corp2inds])
			copyto!(πtemp, settings._zeroer_K1K2);
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
"""
    calc_perp(model,unobs1, obs1, unobs2, obs2, counts, ϕ1, ϕ2)

calculates the perplexity per view.
"""
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


		theta_bar = get_holdout_Θd(obs1_dict, obs2_dict,d, model, count_params,settings)
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
function evaluate_at_epoch(folder,model, h_map, settings, count_params,hos1_dict,obs1_dict,hos2_dict,obs2_dict,perp1_list,perp2_list,VI_CONVERGED, epoch_count)

	ϕ1_est = mean_dir_by_row(model._λ1)
	ϕ2_est = mean_dir_by_row(model._λ2)
	@info "computing perplexity..."
	p1, p2 = calc_perp(model,hos1_dict,obs1_dict,hos2_dict,obs2_dict,
	count_params, ϕ1_est, ϕ2_est, settings)
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

"""
    update_α_newton_mb!(model, ρ, D1, mb, h_map, settings)
uses one step `-H⁻¹g` step towards the current value of α using the learning rate `ρ`
"""

function update_α_newton_mb!(model::MVD,ρ, _D1::Int64,mb::Vector{Int64}, h_map::Vector{Bool}, settings::Settings)
	D = _D1
	n = length(mb)
	logphat = vectorize_mat(mean(dir_expectation2D.(model._γ[mb])))     # <=====  HERE

	K = prod(size(model._α))
	Alpha = vectorize_mat(deepcopy(model._α))
	Alpha_new = zeros(Float64,K)
	ga = zeros(Float64,K)
	Ha = zeros(Float64,K)
	ga .= D .* (logphat .- dir_expectation(Alpha)) ####
	Ha .= -D .*dΨ.(Alpha)   ####
	sumgh = sum(ga./Ha)
	sum1h = sum(1.0./Ha)
	z = D*dΨ(sum(Alpha)) ####
	c = sumgh/(1.0/z + sum1h)
	step_ = ρ .* (ga .- c)./Ha

	if all(Alpha .> step_)
		Alpha_new = Alpha .- step_
		copyto!(model._α, matricize_vec(Alpha_new, model._K1, model._K2))
	end
end
"""
    update_η1_newton_mb!(model, ρ, settings)
uses one step `-H⁻¹g` step towards the current value of η1 using the learning rate `ρ`
"""
function update_η1_newton_mb!(model::MVD,ρ, settings::Settings)
	D = size(model._η1,1)
	logphat = (sum(dir_expectation(model._λ1[k,:]) for k in 1:D) ./ D)    # <=====  HERE
	V = model._corpus1._V
	#Alpha = ones(Float64, 25)
	eta1 = deepcopy(model._η1[1,:])
	eta1_new = zeros(Float64,V)
	ga = zeros(Float64,V)
	Ha = zeros(Float64,V)


	sumgh = 0.0
	sum1h = 0.0
	ga .= D .* (logphat .- dir_expectation(eta1))
	Ha .= -D .* dΨ.(eta1)
	sumgh = sum(ga ./ Ha)
	sum1h = sum(1.0./Ha)
	z = D*dΨ(sum(eta1))
	c = sumgh/(1.0/z + sum1h)

	step_ = ρ .* (ga .- c)./Ha
	if all(eta1 .> step_)
		eta1_new = eta1 .- step_
		copyto!(model._η1, collect(repeat(eta1_new, inner=(1,D))'))
	end
end
"""
    update_η2_newton_mb!(model, ρ, settings)
uses one step `-H⁻¹g` step towards the current value of η2 using the learning rate `ρ`
"""
function update_η2_newton_mb!(model::MVD,ρ,  settings::Settings)
	D = size(model._η2,1)
	logphat = (sum(dir_expectation(model._λ2[k,:]) for k in 1:D) ./ D)    # <=====  HERE
	V = model._corpus2._V
	eta2 = deepcopy(model._η2[1,:])
	eta2_new = zeros(Float64,V)
	ga = zeros(Float64,V)
	Ha = zeros(Float64,V)


	sumgh = 0.0
	sum1h = 0.0
	ga .= D .* (logphat .- dir_expectation(eta2))
	Ha .= -D .* dΨ.(eta2)
	sumgh = sum(ga ./ Ha)
	sum1h = sum(1.0./Ha)
	z = D*dΨ(sum(eta2))
	c = sumgh/(1.0/z + sum1h)

	step_ = ρ .* (ga .- c)./Ha
	if all(eta2 .> step_)
		eta2_new = eta2 .- step_
		copyto!(model._η2, collect(repeat(eta2_new, inner=(1,D))'))
	end
end

# function elbo_at_epoch(model)
#
# end
print("")
