function init_params(K1::Int64,K2::Int64, beta1_prior_, beta2_prior_,
	 alpha_prior_, corpus1_, corpus2_)
	N = max(corpus1_.N, corpus2_.N)

	alpha_vec = rand(Uniform(0.0,alpha_prior_), (K1*K2))
	Alpha = matricize_vec(alpha_vec, K1, K2)
	# B1 = rand(Uniform(0.0, beta1_prior_), (K1, corpus1_.V))
	# B2 = rand(Uniform(0.0, beta2_prior_), (K2, corpus2_.V))
	B1 = collect(repeat(rand(Uniform(0.0, beta1_prior_), corpus1_.V), inner=(1, K1))')
	B2 = collect(repeat(rand(Uniform(0.0, beta2_prior_), corpus2_.V), inner=(1, K2))')
	γ = [ones(Float64, (K1, K2)) for i in 1:N]
	b1 = deepcopy(B1)
	b2 = deepcopy(B2)
	Elog_B1 = zeros(Float64, (K1, corpus1_.V))
	Elog_B2 = zeros(Float64, (K2, corpus2_.V))
	Elog_Theta = [zeros(Float64, (K1, K2)) for i in 1:N]
	zeroer_i = zeros(Float64, (K1, K2))
	zeroer_mb_1 = zeros(Float64, (K1,corpus1_.V))
	zeroer_mb_2 = zeros(Float64, (K2,corpus2_.V))
	sum_phi_1_i = similar(zeroer_i)
	sum_phi_2_i = similar(zeroer_i)
	sum_phi_1_mb = similar(zeroer_mb_1)
	sum_phi_2_mb = similar(zeroer_mb_2)
	old_γ = similar(zeroer_i)
	old_b1 = similar(b1)
	old_b2 = similar(b2)
	old_Alpha = similar(Alpha)
	old_B1 = similar(B1)
	old_B2 = similar(B2)
	temp = similar(zeroer_i)
	sstat_i = similar(zeroer_i)
	sstat_mb_1 = zeros(Float64, K1)
	sstat_mb_2 = zeros(Float64, K2)
	alpha_sstat = [deepcopy(zeroer_i) for i in 1:N]
	return 	Alpha,old_Alpha,B1,old_B1,B2,old_B2,Elog_B1,Elog_B2,Elog_Theta,γ,old_γ,b1,old_b1,b2,old_b2,
	temp,sstat_i,sstat_mb_1,sstat_mb_2,sum_phi_1_mb,sum_phi_2_mb,sum_phi_1_i,sum_phi_2_i, alpha_sstat
end

function epoch_batches(N::Int64, mb_size::Int64, h_map::Vector{Bool})
	N_ = N - sum(h_map)
	div_ = div(N_, mb_size)
	nb = (div_ * mb_size - N_) < 0 ? div_ + 1 : div_
	y = shuffle(collect(1:N)[.!h_map])
	x = [Int64[] for _ in 1:nb]
	for n in 1:nb
		while length(x[n]) < mb_size && !isempty(y)
			push!(x[n],pop!(y))
		end
	end
	return x, nb
end



function setup_hmap(model, h)
	h_count = convert(Int64, floor(h*N))
	corp1 = deepcopy(model.Corpus1)
	corp2 = deepcopy(model.Corpus2)
	while true
		h_map = repeat([false], N)
		inds = sample(1:N, h_count, replace=false, ordered=true)
		h_map[inds] .= true
		x1 = [corp1.docs[i].terms for i in collect(1:corp1.N)[.!h_map]]
		x2 = [corp2.docs[i].terms for i in collect(1:corp2.N)[.!h_map]]
		cond1 = any(.!isempty.(x1)) &&  any(.!isempty.(x2))
		cond2 = (length(unique(vcat(x1...))) == corp1.V) && (length(unique(vcat(x2...))) == corp2.V)
		if cond1 & cond2
			return h_map
		end
	end
end

function split_ho_obs(model, h_map)
	test_ids = findall(h_map)
	hos1_dict = Dict{Int64, Vector{Int64}}()
	obs1_dict = Dict{Int64, Vector{Int64}}()
	hos2_dict = Dict{Int64, Vector{Int64}}()
	obs2_dict = Dict{Int64, Vector{Int64}}()
	for i in test_ids
		if !haskey(hos1_dict, i)
			hos1_dict[i] = getkey(hos1_dict, i, Int64[])
			obs1_dict[i] = getkey(obs1_dict, i, Int64[])
			hos2_dict[i] = getkey(hos2_dict, i, Int64[])
			obs2_dict[i] = getkey(obs2_dict, i, Int64[])
		end
		terms_1 = model.Corpus1.docs[i].terms
		terms_2 = model.Corpus2.docs[i].terms
		partition_1 = div(length(terms_1),10)
		partition_2 = div(length(terms_2),10)
		hos1  = terms_1[1:partition_1]
		obs1  = terms_1[partition_1+1:end]
		hos2  = terms_2[1:partition_2]
		obs2  = terms_2[partition_2+1:end]
		hos1_dict[i] = hos1
		obs1_dict[i] = obs1
		hos2_dict[i] = hos2
		obs2_dict[i] = obs2
	end

	return hos1_dict,obs1_dict,hos2_dict,obs2_dict
end
