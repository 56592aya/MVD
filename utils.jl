################################################################################
##############################  DATA STRUCURES  ################################
################################################################################
VectorList{T} = Vector{Vector{T}}
MatrixList{T} = Vector{Matrix{T}}
mutable struct Document
	terms::Vector{Int64}
	counts::Vector{Int64}
	len::Int64
end
mutable struct Corpus
	docs::Vector{Document}
	N::Int64
	V::Int64
end

struct CountParams
	N::Int64
	N2::Int64
	K1::Int64
	K2::Int64
end

mutable struct MVD
    K1::Int64
    K2::Int64
    Corpus1::Corpus
    Corpus2::Corpus
    Alpha::Matrix{Float64}
    old_Alpha::Matrix{Float64}
    B1::Matrix{Float64}
    old_B1::Matrix{Float64}
	B2::Matrix{Float64}
    old_B2::Matrix{Float64}
    Elog_B1::Matrix{Float64}
    Elog_B2::Matrix{Float64}
    Elog_Theta::MatrixList{Float64}
    γ::MatrixList{Float64}
    old_γ::Matrix{Float64}
    b1::Matrix{Float64}
    old_b1::Matrix{Float64}
    b2::Matrix{Float64}
    old_b2::Matrix{Float64}
	temp::Matrix{Float64}
	sstat_i::Matrix{Float64}
	sstat_mb_1::Vector{Float64}
	sstat_mb_2::Vector{Float64}
	sum_phi_1_mb::Matrix{Float64}
	sum_phi_2_mb::Matrix{Float64}
	sum_phi_1_i::Matrix{Float64}
	sum_phi_2_i::Matrix{Float64}
	# alpha_sstat::MatrixList{Float64}
	function MVD(K1::Int64, K2::Int64, Corpus1::Corpus, Corpus2::Corpus,
		 		alpha_prior_::Float64, beta1_prior_::Float64, beta2_prior_::Float64)
		model = new()
		model.K1 = K1
		model.K2 = K2
		model.Corpus1 = Corpus1
		model.Corpus2 = Corpus2
		model.Alpha = matricize_vec(rand(Uniform(0.0,alpha_prior_), (K1*K2)), K1, K2)
		model.old_Alpha = deepcopy(model.Alpha)
		model.B1 = collect(repeat(rand(Uniform(0.0, beta1_prior_), model.Corpus1.V), inner=(1, K1))')
		model.old_B1 = deepcopy(model.B1)
		model.B2 = collect(repeat(rand(Uniform(0.0, beta2_prior_), model.Corpus2.V), inner=(1, K2))')
		model.old_B2 = deepcopy(model.B2)
		model.Elog_B1 = zeros(Float64, (K1, model.Corpus1.V))
		model.Elog_B2 = zeros(Float64, (K2, model.Corpus2.V))
		model.Elog_Theta = [zeros(Float64, (K1, K2)) for i in 1:model.Corpus1.N]
		model.γ = [ones(Float64, (K1, K2)) for i in 1:model.Corpus1.N]
		model.old_γ = zeros(Float64, (K1, K2))
		model.b1 = deepcopy(model.B1)
		model.old_b1 = deepcopy(model.b1)
		model.b2 = deepcopy(model.B2)
		model.old_b2 = deepcopy(model.b2)
		model.temp = zeros(Float64, (K1, K2))
		model.sstat_i = zeros(Float64, (K1, K2))
		model.sstat_mb_1 = zeros(Float64, K1)
		model.sstat_mb_2 = zeros(Float64, K2)
		model.sum_phi_1_mb = zeros(Float64, (K1,model.Corpus1.V))
		model.sum_phi_2_mb = zeros(Float64, (K1,model.Corpus1.V))
		model.sum_phi_1_i =  zeros(Float64, (K1, K2))
		model.sum_phi_2_i =  zeros(Float64, (K1, K2))
		return model
	end
end

mutable struct Settings
	zeroer_i::Matrix{Float64}
	zeroer_mb_1::Matrix{Float64}
	zeroer_mb_2::Matrix{Float64}
	MAX_VI_ITER::Int64
	MAX_ALPHA_ITER::Int64
	MAX_GAMMA_ITER::Int64
	MAX_ALPHA_DECAY::Int64
	ALPHA_DECAY_FACTOR::Float64
	ALPHA_THRESHOLD::Float64
	GAMMA_THRESHOLD::Float64
	VI_THRESHOLD::Float64
	EVAL_EVERY::Int64
	LR_OFFSET::Float64
	LR_KAPPA::Float64
	function Settings(K1::Int64, K2::Int64,Corpus1::Corpus, Corpus2::Corpus,
					 MAX_VI_ITER::Int64,MAX_ALPHA_ITER::Int64,MAX_GAMMA_ITER::Int64,
					 MAX_ALPHA_DECAY::Int64,ALPHA_DECAY_FACTOR::Float64,
					 ALPHA_THRESHOLD::Float64,GAMMA_THRESHOLD::Float64,
					 VI_THRESHOLD::Float64,EVAL_EVERY::Int64,LR_OFFSET::Float64,
					 LR_KAPPA::Float64)
		settings = new()
		settings.zeroer_i = zeros(Float64, (K1, K2))
		settings.zeroer_mb_1 = zeros(Float64, (K1,Corpus1.V))
		settings.zeroer_mb_2 = zeros(Float64, (K2,Corpus2.V))
		settings.MAX_VI_ITER = MAX_VI_ITER
		settings.MAX_ALPHA_ITER = MAX_ALPHA_ITER
		settings.MAX_GAMMA_ITER = MAX_GAMMA_ITER
		settings.MAX_ALPHA_DECAY = MAX_ALPHA_DECAY
		settings.ALPHA_DECAY_FACTOR = ALPHA_DECAY_FACTOR
		settings.ALPHA_THRESHOLD = ALPHA_THRESHOLD
		settings.GAMMA_THRESHOLD = GAMMA_THRESHOLD
		settings.VI_THRESHOLD = VI_THRESHOLD
		settings.EVAL_EVERY = EVAL_EVERY
		settings.LR_OFFSET = LR_OFFSET
		settings.LR_KAPPA = LR_KAPPA
		return settings
	end
end
################################################################################
##############################  Uitlity Funcs   ################################
################################################################################
function digamma_(x::Float64)
  	x=x+6.0
  	p=1.0/abs2(x)
  	p= (((0.004166666666667*p-0.003968253986254)*p+0.008333333333333)*p-0.083333333333333)*p
  	p= p+log(x)-0.5/x-1.0/(x-1.0)-1.0/(x-2.0)-1.0/(x-3.0)-1.0/(x-4.0)-1.0/(x-5.0)-1.0/(x-6.0)
  	p
end

function trigamma_(x::Float64)
    x=x+6.0;
    p=1.0/(x*x);
    p=(((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238)
         *p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p;
    for i in 1:6
        x=x-1.0;
        p=1.0/(x*x)+p;
	end
    return(p)
end

function vectorize_mat(mat::Matrix{Float64})
	K1, K2 = size(mat)
	vec_ = zeros(Float64, prod(size(mat)))
	for k1 in 1:K1
		for k2 in 1:K2
			vec_[K2*(k1-1) + k2] = mat[k1, k2]
		end
	end
	return vec_
end
function vectorize_mat(mat::Matrix{Int64})
	K1, K2 = size(mat)
	vec_ = zeros(Float64, prod(size(mat)))
	for k1 in 1:K1
		for k2 in 1:K2
			vec_[K2*(k1-1) + k2] = mat[k1, k2]
		end
	end
	return vec_
end
function matricize_vec(vec_::Vector{Float64}, K1::Int64, K2::Int64)
	mat_ = zeros(Float64, (K1, K2))
	for j in 1:length(vec_)
		m = ceil(Int64, j/K2)
		mat_[m, j-((m-1)*K2)] = vec_[j]
	end
	return mat_
end
function matricize_vec(vec_::Vector{Int64}, K1::Int64, K2::Int64)
	mat_ = zeros(Float64, (K1, K2))
	for j in 1:length(vec_)
		m = ceil(Int64, j/K2)
		mat_[m, j-((m-1)*K2)] = vec_[j]
	end
	return mat_
end


function δ(i::Int64,j::Int64)
	if i == j
		return 1
	else
		return 0
	end
end


function Elog(Mat::Matrix{Float64})
    digamma_.(Mat) .- digamma_(sum(Mat))
end

# function Elog(Vec::Vector{Float64})
#     digamma_.(Vec) .- digamma_(sum(Vec))
# end
function Elog(Vec::AbstractVector{Float64})
    digamma_.(Vec) .- digamma_(sum(Vec))
end
function logsumexp(X::Vector{Float64})

    alpha = -Inf::Float64;
	r = 0.0;
    @inbounds for x in X
        if x <= alpha
            r += exp(x - alpha)
        else
            r *= exp(alpha - x)
            r += 1.0
            alpha = x
        end
    end
    log(r) + alpha
end
function logsumexp(X::Matrix{Float64})
    alpha = -Inf::Float64;
	r = 0.0;
    @inbounds for x in X
        if x <= alpha
            r += exp(x - alpha)
        else
            r *= exp(alpha - x)
            r += 1.0
            alpha = x
        end
    end
    log(r) + alpha
end


function logsumexp(X::Float64, Y::Float64)
    alpha = -Inf::Float64;
	r = 0.0;
    if X <= alpha
        r += exp(X - alpha)
    else
        r *= exp(alpha - X)
        r += 1.0
        alpha = X
    end
    if Y <= alpha
        r += exp(Y - alpha)
    else
        r *= exp(alpha - Y)
        r += 1.0
        alpha = Y
    end
    log(r) + alpha
end

function softmax!(MEM::Matrix{Float64},X::Matrix{Float64})
    lse = logsumexp(X)
    @. (MEM = (exp(X - lse)));
end
function softmax!(MEM::Matrix{Float64})
    softmax!(MEM, MEM)
end
function softmax(X::Matrix{Float64})
    return exp.(X .- logsumexp(X))
end

function sort_by_argmax!(X::Matrix{Float64})

	n_row=size(X,1)
	n_col = size(X,2)
	ind_max=zeros(Int64, n_row)
	permuted_index = zeros(Int64, n_row)
	for a in 1:n_row
    	ind_max[a] = findmax(view(X,a,1:n_col))[2]
	end
	X_tmp = similar(X)
	count_ = 1
	for j in 1:maximum(ind_max)
  		for i in 1:n_row
    		if ind_max[i] == j
	      		for k in 1:n_col
	        		X_tmp[count_, k] = X[i,k]
	      		end
				permuted_index[count_]=i
      			count_ += 1
    		end
  		end
	end
	X[:]=X_tmp[:]
	X, permuted_index
end

function find_all(val::Int64, doc::Vector{Int64})
	findall(x -> x == val, doc)
end
function get_lr(i::Int64, settings::Settings)
	return (settings.LR_OFFSET+convert(Float64, i))^(-settings.LR_KAPPA)
end
function mean_change(new::AbstractArray{R}, old::AbstractArray{R}) where  {R<:AbstractFloat}
	n = length(new)
	change = sum(abs.(new .- old))/n
	return(change)
end


function add_vec2mcols!(mem::AbstractArray{R},mat::AbstractArray{R}, vec::AbstractVector{T}) where {R<:AbstractFloat,T<:Real}
	@inbounds for I in CartesianIndices(mat)
		mem[I] = mat[I] + vec[I.I[1]]
	end

end

function add_vec2mrows!(mem::AbstractArray{R},mat::AbstractArray{R}, vec::AbstractVector{T}) where {R<:AbstractFloat,T<:Real}
	@inbounds for I in CartesianIndices(mat)
		mem[I] = mat[I] + vec[I.I[2]]
	end
end
function rowsums!(mem::AbstractVector{R}, mat::AbstractArray{R}) where {R<:AbstractFloat,T<:Real}
	@inbounds for I in CartesianIndices(mat)
		mem[I.I[1]] += mat[I]
	end
end

function colsums!(mem::AbstractVector{R}, mat::AbstractArray{R}) where {R<:AbstractFloat,T<:Real}
	@inbounds for I in CartesianIndices(mat)
		mem[I.I[2]] += mat[I]
	end
end
################################################################################
##############################  Data Wrangling  ################################
################################################################################
function fix_corp!(model::MVD, folder::String)
	c1 = deepcopy(model.Corpus1)
	for i in 1:length(model.Corpus1.docs)
		doc1 = model.Corpus1.docs[i]
		uniqs1 = unique(doc1.terms)
		counts1 = Int64[]
		for u in uniqs1
			counts1 = vcat(counts1, length(find_all(u, doc1.terms)))
		end
		c1.docs[i] = Document(uniqs1,counts1,doc1.len)

	end
	c2 = deepcopy(model.Corpus2)
	for i in 1:length(model.Corpus2.docs)
		doc2 = model.Corpus2.docs[i]
		uniqs2 = unique(doc2.terms)
		counts2 = Int64[]
		for u in uniqs2
			counts2 = vcat(counts2, length(find_all(u, doc2.terms)))
		end
		c2.docs[i] = Document(uniqs2,counts2,doc2.len)
	end
	model.Corpus1 = c1
	model.Corpus2 = c2
	corp1 = deepcopy(model.Corpus1)
	corp2 = deepcopy(model.Corpus2)
	@save "$(folder)/Corpus1" corp1
	@save "$(folder)/Corpus2" corp2
end

function figure_sparsity!(model::MVD, sparsity::Float64, all_::Bool, folder::String)
	while true
		y2 = Int64[]
		if all_
			if sparsity == 0.0
				break
			end
			y2 = Int64[]
			corp = deepcopy(model.Corpus2)
			for i in 1:length(corp.docs)
				num_remove = sparsity*corp.docs[i].len
				count = 0
				it = 1
				rs = shuffle(collect(1:length(corp.docs[i].terms)))
				to_remove = Int64[]
				while count < num_remove
					count += corp.docs[i].counts[rs[it]]
					it +=1
					to_remove = vcat(to_remove, rs[it])
				end
				corp.docs[i].counts[to_remove] .= 0
				corp.docs[i].len = sum(corp.docs[i].counts)
				y2 = unique(vcat(y2, corp.docs[i].terms[corp.docs[i].counts .>0]))
			end
			if length(unique(y2)) == corp.V
				model.Corpus2 = corp
				break
			else
				y2 = Int64[]
			end

		else
			if sparsity == 0.0
				break
			end
			y2 = Int64[]
			corp = deepcopy(model.Corpus2)
			num_remove = floor(Int64, sparsity*length(corp.docs))
			to_remove = sample(collect(1:length(corp.docs)), num_remove, replace=false)
			for j in 1:length(corp.docs)
				if j in to_remove
					corp.docs[j].counts .= 0
					corp.docs[j].len = sum(corp.docs[j].counts)
				else
					y2 = unique(vcat(y2, corp.docs[j].terms))
				end
			end
			if length(unique(y2)) == corp.V
				model.Corpus2 = corp
				break
			else
				y2 = Int64[]
			end
		end
	end
	corp2_sparse = deepcopy(model.Corpus2)
	@save "$(folder)/Corpus2_sparse" corp2_sparse
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



function setup_hmap(model::MVD, h::Float64,N::Int64)
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

function split_ho_obs(model::MVD, h_map::Vector{Bool})
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
