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
	alpha_sstat::MatrixList{Float64}
end

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


function δ(i,j)
	if i == j
		return 1
	else
		return 0
	end
end


function Elog(Mat::Matrix{Float64})
    digamma_.(Mat) .- digamma_(sum(Mat))
end

function Elog(Vec::Vector{Float64})
    digamma_.(Vec) .- digamma_(sum(Vec))
end

function Elog(Vec)
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
function softmax(X::Matrix{Float64})
    return exp.(X .- logsumexp(X))
end
function logsumexp(X)
    isempty(X) && return log(sum(X))
    reduce(logaddexp, X)
end
function logsumexp(X::AbstractArray{T}; dims=:) where {T<:Real}
    u = reduce(max, X, dims=dims, init=log(zero(T)))
    u isa AbstractArray || isfinite(u) || return float(u)
    let u=u
        if u isa AbstractArray
            u .+ log.(sum(exp.(X .- u); dims=dims))
        else
            u + log(sum(x -> exp(x-u), X))
        end
    end
end
function softmax!(r::AbstractArray{R}, x::AbstractArray{T}) where {R<:AbstractFloat,T<:Real}
    n = length(x)
    length(r) == n || throw(DimensionMismatch("Inconsistent array lengths."))
    u = maximum(x)
    s = 0.
    @inbounds for i = 1:n
        s += (r[i] = exp(x[i] - u))
    end
    invs = convert(R, inv(s))
    @inbounds for i = 1:n
        r[i] *= invs
    end
    r
end
softmax!(x::AbstractArray{<:AbstractFloat}) = softmax!(x, x)
softmax(x::AbstractArray{<:Real}) = softmax!(similar(x, Float64), x)

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

function find_all(val, doc)
	findall(x -> x == val, doc)
end
function get_lr(epoch::Int64, S::Float64, κ::Float64)
	return (S+convert(Float64, epoch))^(-κ)
end
function mean_change(new::AbstractArray{R}, old::AbstractArray{R}) where  {R<:AbstractFloat}
	n = length(new)
	change = sum(abs.(new .- old))/n
	return(change)
end
function fix_corp!(model)
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
end

function figure_sparsity!(model, sparsity, all_)
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
