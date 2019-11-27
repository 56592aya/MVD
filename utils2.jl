################################################################################
##############################  DATA STRUCURES  ################################
################################################################################
VectorList{T} = AbstractVector{AbstractVector{T}}
MatrixList{T} = AbstractVector{AbstractArray{T,2}}

function vectorize_mat(mat::T)  where T <:AbstractArray{Float64}
	K1, K2 = size(mat)
	vec_ = zeros(Float64, prod(size(mat)))
	for k1 in 1:K1
		for k2 in 1:K2
			vec_[K2*(k1-1) + k2] = mat[k1, k2]
		end
	end
	return vec_
end
function get1DColIndex(K1, K2, c)
	return [K2*(k1-1) + c for k1 in 1:K1]
end
function get1DColIndices(K1, K2)
	collect(hcat([[K2*(k1-1) + c for k1 in 1:K1] for c in 1:K2]...)')
end

function get1DRowIndex(K1, K2, r)
	return [K2*(r-1) + k2 for k2 in 1:K2]
end
function get1DRowIndices(K1, K2)
	return collect(hcat([[K2*(r-1) + k2 for k2 in 1:K2] for r in 1:K1]...)')
end


function vectorize_mat(mat::T) where T <:AbstractArray{Int64}
	K1, K2 = size(mat)
	vec_ = zeros(Float64, prod(size(mat)))
	for k1 in 1:K1
		for k2 in 1:K2
			vec_[K2*(k1-1) + k2] = mat[k1, k2]
		end
	end
	return vec_
end


function matricize_vec(vec_::T, K1::Int64, K2::Int64) where T <: AbstractVector{Int64}
	mat_ = zeros(Float64, (K1, K2))
	for j in 1:length(vec_)
		m = ceil(Int64, j/K2)
		mat_[m, j-((m-1)*K2)] = vec_[j]
	end
	return mat_
end
function matricize_vec(vec_::T, K1::Int64, K2::Int64) where T <: AbstractVector{Float64}
	mat_ = zeros(Float64, (K1, K2))
	for j in 1:length(vec_)
		m = ceil(Int64, j/K2)
		mat_[m, j-((m-1)*K2)] = vec_[j]
	end
	return mat_
end

mutable struct Document <: AbstractDocument
	_terms::AbstractVector{Int64}
	_counts::AbstractVector{Int64}
	_length::Int64
end

mutable struct Corpus <: AbstractCorpus
	_docs::AbstractVector{Document}
	_D::Int64
	_V::Int64
end

struct TrainCounts <: AbstractCounts
	_D1::Int64
	_D2::Int64
	_K1::Int64
	_K2::Int64
end

mutable struct MVD <: AbstractModel
    _K1::Int64
    _K2::Int64
    _corpus1::Corpus
    _corpus2::Corpus
    _α::AbstractArray{Float64}
    _α_old::AbstractArray{Float64}
    _η1::AbstractArray{Float64}
    _η1_old::AbstractArray{Float64}
	_η2::AbstractArray{Float64}
    _η2_old::AbstractArray{Float64}
    _elogϕ1::AbstractArray{Float64}
    # _expelogϕ1::AbstractArray{Float64}
    _elogϕ2::AbstractArray{Float64}
    # _expelogϕ2::AbstractArray{Float64}
    # _elogΘ::AbstractVector{Matrix{Float64}}
    # _expelogΘ::AbstractVector{Matrix{Float64}}
    _γ::AbstractVector{Matrix{Float64}}
    _γ_old::AbstractArray{Float64}
    _λ1::AbstractArray{Float64}
    _λ1_old::AbstractArray{Float64}
    _λ2::AbstractArray{Float64}
    _λ2_old::AbstractArray{Float64}
	# _πtemp::AbstractArray{Float64}
	# _sstat_i::AbstractArray{Float64}
	# _sstat1_mb::AbstractVector{Float64}
	# _sstat2_mb::AbstractVector{Float64}
	_sumπ1_mb::AbstractArray{Float64}
	_sumπ2_mb::AbstractArray{Float64}
	# _sumπ1_i::AbstractArray{Float64}
	# _sumπ2_i::AbstractArray{Float64}

	function MVD(K1::Int64, K2::Int64, corpus1::Corpus, corpus2::Corpus,
		 		alpha_prior_::Float64, eta1_prior_::Float64, eta2_prior_::Float64)
		model = new()
		model._K1 = K1
		model._K2 = K2
		model._corpus1 = corpus1
		model._corpus2 = corpus2
		model._α = matricize_vec(rand(Uniform(0.0,alpha_prior_), (K1*K2)), K1, K2)
		model._α_old = deepcopy(model._α)
		model._η1 = collect(repeat(rand(Uniform(0.0, eta1_prior_), model._corpus1._V), inner=(1, K1))')
		model._η1_old = deepcopy(model._η1)
		model._η2 = collect(repeat(rand(Uniform(0.0, eta2_prior_), model._corpus2._V), inner=(1, K2))')
		model._η2_old = deepcopy(model._η2)
		model._elogϕ1 = zeros(Float64, (K1, model._corpus1._V))
		# model._expelogϕ1 = zeros(Float64, (K1, model._corpus1._V))
		model._elogϕ2 = zeros(Float64, (K2, model._corpus2._V))
		# model._expelogϕ2 = zeros(Float64, (K2, model._corpus2._V))
		# model._elogΘ = [zeros(Float64, (K1, K2) for _ in 1:model._corpus1._D)]
		# model._expelogΘ = [zeros(Float64, (K1, K2) for _ in 1:model._corpus1._D)]
		model._γ = [zeros(Float64, (K1, K2)) for _ in 1:model._corpus1._D]
		model._γ_old = zeros(Float64, (K1, K2))
		model._λ1 = deepcopy(model._η1)
		model._λ1_old = deepcopy(model._λ1)
		model._λ2 = deepcopy(model._η2)
		model._λ2_old = deepcopy(model._λ2)
		# model._πtemp = zeros(Float64, (K1, K2))
		# model._sstat_i = zeros(Float64, (K1, K2))
		# model._sstat1_mb = zeros(Float64, K1)
		# model._sstat2_mb = zeros(Float64, K2)
		model._sumπ1_mb = zeros(Float64, (K1,model._corpus1._V))
		model._sumπ2_mb = zeros(Float64, (K1,model._corpus2._V))
		# model._sumπ1_i =  zeros(Float64, (K1, K2))
		# model._sumπ2_i =  zeros(Float64, (K1, K2))
		return model
	end
end

mutable struct Settings <: AbstractSettings
	_zeroer_K1K2::AbstractArray{Float64}
	_zeroer_K1V1::AbstractArray{Float64}
	_zeroer_K2V2::AbstractArray{Float64}
	_MAX_VI_ITER::Int64
	_MAX_ALPHA_ITER::Int64
	_MAX_GAMMA_ITER::Int64
	_MAX_ALPHA_DECAY::Int64
	_ALPHA_DECAY_FACTOR::Float64
	_ALPHA_THRESHOLD::Float64
	_GAMMA_THRESHOLD::Float64
	_VI_THRESHOLD::Float64
	_EVAL_EVERY::Int64
	_LR_OFFSET::Float64
	_LR_KAPPA::Float64
	function Settings(K1::Int64, K2::Int64,corpus1::Corpus, corpus2::Corpus,
					 MAX_VI_ITER::Int64,MAX_ALPHA_ITER::Int64,MAX_GAMMA_ITER::Int64,
					 MAX_ALPHA_DECAY::Int64,ALPHA_DECAY_FACTOR::Float64,
					 ALPHA_THRESHOLD::Float64,GAMMA_THRESHOLD::Float64,
					 VI_THRESHOLD::Float64,EVAL_EVERY::Int64,LR_OFFSET::Float64,
					 LR_KAPPA::Float64)
		settings = new()
		settings._zeroer_K1K2 = zeros(Float64, (K1, K2))
		settings._zeroer_K1V1 = zeros(Float64, (K1,corpus1._V))
		settings._zeroer_K2V2 = zeros(Float64, (K2,corpus2._V))
		settings._MAX_VI_ITER = MAX_VI_ITER
		settings._MAX_ALPHA_ITER = MAX_ALPHA_ITER
		settings._MAX_GAMMA_ITER = MAX_GAMMA_ITER
		settings._MAX_ALPHA_DECAY = MAX_ALPHA_DECAY
		settings._ALPHA_DECAY_FACTOR = ALPHA_DECAY_FACTOR
		settings._ALPHA_THRESHOLD = ALPHA_THRESHOLD
		settings._GAMMA_THRESHOLD = GAMMA_THRESHOLD
		settings._VI_THRESHOLD = VI_THRESHOLD
		settings._EVAL_EVERY = EVAL_EVERY
		settings._LR_OFFSET = LR_OFFSET
		settings._LR_KAPPA = LR_KAPPA
		return settings
	end
end
################################################################################
##############################  Uitlity Funcs   ################################
################################################################################

function expdot(X::AbstractVector{Matrix{Float64}})
	[exp.(x) for x in X]
end
function Ψ(x::T) where T<: Real
    ψ_ = 0.0
    if x < 7.0
        # shift using recurrence formula
        n = 7 - floor(Int,x)
        for ν = 1:n-1
            ψ_ -= 1.0/(x + ν)
        end
        ψ_ -= 1.0/x
        x += n
    end
    t = 1.0/x
    ψ_ += log(x) - 0.5*t
    t *= t # 1/z^2
    # the coefficients here are Float64(bernoulli[2:9] .// (2*(1:8)))
    ψ_ -= t * @evalpoly(t,0.08333333333333333,-0.008333333333333333,0.003968253968253968,-0.004166666666666667,0.007575757575757576,-0.021092796092796094,0.08333333333333333,-0.4432598039215686)
end


function dΨ(x::T) where T <: Real
    # if x <= 0 # reflection formula
    #     return (π * csc(π*x))^2 - dΨ(1 - x)
    # end
    ψ_ = 0.0
    if x < 8.0
        # shift using recurrence formula
        n = 8 - floor(Int,x)
        ψ_ += inv(x)^2.0
        for ν = 1:n-1
            ψ_ += inv(x + ν)^2.0
        end
        x += n
    end
    t = inv(x)
    w = t * t # 1/z^2
    ψ_ += t + 0.5*w
    # the coefficients here are Float64(bernoulli[2:9])
    ψ_ += t*w * @evalpoly(w,0.16666666666666666,-0.03333333333333333,0.023809523809523808,-0.03333333333333333,0.07575757575757576,-0.2531135531135531,1.1666666666666667,-7.092156862745098)
end



function δ(i::Int64,j::Int64)
	if i == j
		return 1
	else
		return 0
	end
end

function dir_expectation(X::T) where T <: AbstractVector{Float64}
    Ψ.(X) .- Ψ(sum(X))
end
function dir_expectation!(Y::T,X::T) where T <: AbstractVector{Float64}
    Y .= dir_expectation(X)
end
function dir_expectation2D(X::T) where T <: AbstractArray{Float64}
    Ψ.(X) .- Ψ(sum(X))
end
function dir_expectation2D!(Y::T,X::T) where T <: AbstractArray{Float64}
	Y .= dir_expectation2D(X)
end


function dir_expectationByRow!(Y::T,X::T) where T <: AbstractArray{Float64}
	for (k,row) in enumerate(eachrow(X))
		Y[k,:] .= dir_expectation(row)
	end
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
function get_ρ(i::Int64, settings::Settings)
	return (settings._LR_OFFSET+convert(Float64, i))^(-settings._LR_KAPPA)
end
function get_ρ(epoch_count, mb, mindex,settings::Settings)
	return (settings._LR_OFFSET+epoch_count+(mindex/length(mb)))^(-settings._LR_KAPPA)
end
function mean_change(new::AbstractArray{R}, old::AbstractArray{R}) where  {R<:AbstractFloat}
	n = length(new)
	change = sum(abs.(new .- old))/n
	return(change)
end



################################################################################
##############################  Data Wrangling  ################################
################################################################################
function fix_corp!(model::MVD, folder::String)
	c1 = deepcopy(model._corpus1)
	for i in 1:length(model._corpus1._docs)
		doc1 = model._corpus1._docs[i]
		uniqs1 = unique(doc1._terms)
		counts1 = Int64[]
		for u in uniqs1
			counts1 = vcat(counts1, length(find_all(u, doc1._terms)))
		end
		c1._docs[i] = Document(uniqs1,counts1,doc1._length)

	end
	c2 = deepcopy(model._corpus2)
	for i in 1:length(model._corpus2._docs)
		doc2 = model._corpus2._docs[i]
		uniqs2 = unique(doc2._terms)
		counts2 = Int64[]
		for u in uniqs2
			counts2 = vcat(counts2, length(find_all(u, doc2._terms)))
		end
		c2._docs[i] = Document(uniqs2,counts2,doc2._length)
	end
	model._corpus1 = c1
	model._corpus2 = c2
	corp1 = deepcopy(model._corpus1)
	corp2 = deepcopy(model._corpus2)
	@save "$(folder)/corpus1" corp1
	@save "$(folder)/corpus2" corp2
end

function figure_sparsity!(model::MVD, sparsity::Float64, all_::Bool, folder::String)
	while true
		y2 = Int64[]
		if all_
			if sparsity == 0.0
				break
			end
			y2 = Int64[]
			corp = deepcopy(model._corpus2)
			for i in 1:length(corp._cdocs)
				num_remove = sparsity*corp._docs[i]._length
				count = 0
				it = 1
				rs = shuffle(collect(1:length(corp._docs[i]._terms)))
				to_remove = Int64[]
				while count < num_remove
					count += corp._docs[i]._counts[rs[it]]
					it +=1
					to_remove = vcat(to_remove, rs[it])
				end
				corp._docs[i]._counts[to_remove] .= 0
				corp._docs[i]._length = sum(corp._docs[i]._counts)
				y2 = unique(vcat(y2, corp._docs[i]._terms[corp._docs[i]._counts .>0]))
			end
			if length(unique(y2)) == corp._V
				model._corpus2 = corp
				break
			else
				y2 = Int64[]
			end

		else
			if sparsity == 0.0
				break
			end
			y2 = Int64[]
			corp = deepcopy(model._corpus2)
			num_remove = floor(Int64, sparsity*length(corp._docs))
			to_remove = sample(collect(1:length(corp._docs)), num_remove, replace=false)
			for j in 1:length(corp._docs)
				if j in to_remove
					corp._docs[j]._counts .= 0
					corp._docs[j]._length = sum(corp._docs[j]._counts)
				else
					y2 = unique(vcat(y2, corp._docs[j]._terms))
				end
			end
			if length(unique(y2)) == corp._V
				model._corpus2 = corp
				break
			else
				y2 = Int64[]
			end
		end
	end
	corp2_sparse = deepcopy(model._corpus2)
	@save "$(folder)/corpus2_sparse" corp2_sparse
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
	corp1 = deepcopy(model._corpus1)
	corp2 = deepcopy(model._corpus2)
	while true
		h_map = repeat([false], N)
		inds = sample(1:N, h_count, replace=false, ordered=true)
		h_map[inds] .= true
		x1 = [corp1._docs[i]._terms for i in collect(1:corp1._D)[.!h_map]]
		x2 = [corp2._docs[i]._terms for i in collect(1:corp2._D)[.!h_map]]
		cond1 = any(.!isempty.(x1)) &&  any(.!isempty.(x2))
		cond2 = (length(unique(vcat(x1...))) == corp1._V) && (length(unique(vcat(x2...))) == corp2._V)
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
		terms_1 = model._corpus1._docs[i]._terms
		terms_2 = model._corpus2._docs[i]._terms
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
