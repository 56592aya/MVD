################################################################################
##############################  DATA STRUCURES  ################################
################################################################################
"""
    vectorize_mat(mat)
	 to make a vector from a matrix on the row basis.
	# Examples
	```julia-repl
	julia> mat = [1.0 3.0; 2.0 4.0];mat
	2×2 Array{Float64,2}:
	 1.0  2.0
	 3.0  4.0
	julia> vectorize_mat(mat)
	[1.0,2.0,3.0,4.0]
	```
"""
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
"""
    vectorize_mat(mat)

	Used to make a vector from a matrix on the row basis.
	# Examples
	```julia-repl
	julia> mat = [1 2; 3 4];mat
	2×2 Array{Int64,2}:
	 1  2
	 3  4
	julia> vectorize_mat(mat)
	[1,2,3,4]
	```
"""
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

"""
    matricize_vec(vec, M,N)

	Used to make a matrix from a vector on the row basis given by `M` \n
	rows and `N` columns.
	# Examples
	```julia-repl
	julia> vec = [1.0,2.0,3.0,4.0]
	julia> matricize_vec(vec, 2, 2)
	2×2 Array{Float64,2}:
	 1.0  2.0
	 3.0  4.0
	```
"""
function matricize_vec(vec_::T, K1::Int64, K2::Int64) where T <: AbstractVector{Float64}
	mat_ = zeros(Float64, (K1, K2))
	for j in 1:length(vec_)
		m = ceil(Int64, j/K2)
		mat_[m, j-((m-1)*K2)] = vec_[j]
	end
	return mat_
end
"""
    matricize_vec(vec, M,N)

	Used to make a matrix from a vector on the row basis given by `M` \n
	rows and `N` columns.
	# Examples
	```julia-repl
	julia> vec = [1,2,3,4]
	julia> matricize_vec(vec, 2, 2)
	2×2 Array{Int64,2}:
	 1  2
	 3  4
	```
"""
function matricize_vec(vec_::T, K1::Int64, K2::Int64) where T <: AbstractVector{Int64}
	mat_ = zeros(Float64, (K1, K2))
	for j in 1:length(vec_)
		m = ceil(Int64, j/K2)
		mat_[m, j-((m-1)*K2)] = vec_[j]
	end
	return mat_
end
"""
	Document
	mutable struct for a single document
"""
mutable struct Document <: AbstractDocument
	"terms of a document"
	_terms::AbstractVector{Int64}
	"counts of each term in the order of terms"
	_counts::AbstractVector{Int64}
	"a single number which is the sum of _counts"
	_length::Int64
end

"""
	Corpus
	mutable struct for _corpus of _docs
"""
mutable struct Corpus <: AbstractCorpus
	"list of document"
	_docs::AbstractVector{Document}
	"number of documents"
	_D::Int64
	"number of unique words in the _corpus vocabulary"
	_V::Int64
end

"""
	TrainCounts
	counts that are specific to the training sample
"""
struct TrainCounts <: AbstractCounts
	"number of documents in the 1st view"
	_D1::Int64
	"number of documents in the 2nd view"
	_D2::Int64
	"number of topics in the 1st view"
	_K1::Int64
	"number of topics in the 2nd view"
	_K2::Int64
end
"""
	MVD
	Multivariate Dirichlet model\n
	Consists of constants, variational params and placeholders
"""
mutable struct MVD <: AbstractModel
	"number of topics in the 1st view"
    _K1::Int64
	"number of topics in the 2nd view"
    _K2::Int64
	"corpus of type Corpus for 1st view"
    _corpus1::Corpus
	"corpus of type Corpus for 2nd view"
    _corpus2::Corpus
	"model prior for document-level topic distribution"
    _α::AbstractArray{Float64}
	"old value for document-level topic distribution"
    _α_old::AbstractArray{Float64}
	"model prior for per topic word distribution in the 1st view"
    _η1::AbstractArray{Float64}
	"old value for per topic word distribution in the 1st view"
    _η1_old::AbstractArray{Float64}
	"model prior for per topic word distribution in the 2nd view"
	_η2::AbstractArray{Float64}
	"old value for per topic word distribution in the 2nd view"
    _η2_old::AbstractArray{Float64}
	"variational expectation of log-topics in the 1st view"
    _elogϕ1::AbstractArray{Float64}
	"variational expectation of log-topics in the 2nd view"
    _elogϕ2::AbstractArray{Float64}
	"variational parameter of document-level topic distribution"
    _γ::AbstractVector{Matrix{Float64}}
	"old value for variational parameter of document level topic distribution"
    _γ_old::AbstractArray{Float64}
	"variational parameter for topics in the 1st view"
    _λ1::AbstractArray{Float64}
	"old value for variational parameter for topics in the 1st view"
    _λ1_old::AbstractArray{Float64}
	"variational parameter for topics in the 2nd view"
    _λ2::AbstractArray{Float64}
	"old value variational parameter for topics in the 2nd view"
    _λ2_old::AbstractArray{Float64}
	"psuedocounts for topic contribution of words per minibatch in the 1st view"
	_sumπ1_mb::AbstractArray{Float64}
	"psuedocounts for topic contribution of words per minibatch in the 2nd view"
	_sumπ2_mb::AbstractArray{Float64}

	"""
		MVD(K1, K2, C1, C2, α,η1, η2)
		Used to create a new MVD object
	"""
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
		model._elogϕ2 = zeros(Float64, (K2, model._corpus2._V))
		model._γ = [zeros(Float64, (K1, K2)) for _ in 1:model._corpus1._D]
		model._γ_old = zeros(Float64, (K1, K2))
		model._λ1 = deepcopy(model._η1)
		model._λ1_old = deepcopy(model._λ1)
		model._λ2 = deepcopy(model._η2)
		model._λ2_old = deepcopy(model._λ2)
		model._sumπ1_mb = zeros(Float64, (K1,model._corpus1._V))
		model._sumπ2_mb = zeros(Float64, (K2,model._corpus2._V))
		return model
	end
end

"""
	Settings
	Settings needed for flags and condition checks of inference,\n
	consisting of mostly constants.
"""
mutable struct Settings <: AbstractSettings
	"K1×K2 zero matrix "
	_zeroer_K1K2::AbstractArray{Float64}
	"K1×V1 zero matrix "
	_zeroer_K1V1::AbstractArray{Float64}
	"K2×V2 zero matrix "
	_zeroer_K2V2::AbstractArray{Float64}
	"maximum number of iterations for Variationl Inference"
	_MAX_VI_ITER::Int64
	"maximum number of newton updates for α update;used only in full-batch setting"
	_MAX_ALPHA_ITER::Int64
	"maximum number of iterations for updating γ"
	_MAX_GAMMA_ITER::Int64
	"maximum number of times to decay the learning rate of α;used only in full-batch setting"
	_MAX_ALPHA_DECAY::Int64
	"factor of decay for the learning rate of α;used only in full-batch setting"
	_ALPHA_DECAY_FACTOR::Float64
	"threshold of convergence for update of α"
	_ALPHA_THRESHOLD::Float64
	"threshold of convergence for update of γ"
	_GAMMA_THRESHOLD::Float64
	"threshold of convergence for variational perplexity"
	_VI_THRESHOLD::Float64
	"number of epochs to evaluate perplexity and print status"
	_EVAL_EVERY::Int64
	"offset of learning rate for the global variational parameter"
	_LR_OFFSET::Float64
	"kappa of learning  for the global variational parameter"
	_LR_KAPPA::Float64
	"""
		Settings(K1, K2, C1, C2, VI_I, α_I,γ_I, α_decay,decay_fac, αTh,γTh,vTh,every,offset,κ )
		create a new object of type Settings
	"""
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
"""
	expdot(X)
	vectorized `exp` for the vector of matrices
	# Examples
	```julia-repl
	julia> X = [[1.0 2.0; 3.0 4.0],[5.0 6.0; 7.0 8.0]]
	julia> expdot(X)
	2-element Array{Array{Float64,2},1}:
	 [2.718281828459045 7.38905609893065; 20.085536923187668 54.598150033144236]
	 [148.4131591025766 403.4287934927351; 1096.6331584284585 2980.9579870417283]
	```
"""
function expdot(X::AbstractVector{Matrix{Float64}})
	[exp.(x) for x in X]
end
"""
	lnΓ(x)
	computes the loggamma of `x`
	# Examples
	```julia-repl
	julia> x = 1.0
	julia> lnΓ(x)
	-0.0003604287667536843
	```
"""

function lnΓ(x::T) where T <: Real
	z=1.0 / (x * x);
	x=x + 6.0;
    z =(((-0.000595238095238*z+0.000793650793651)*z-0.002777777777778)*z+0.083333333333333)/x;
    z=(x-0.5)*log(x)-x+0.918938533204673+z-log(x-1)- log(x-2)-log(x-3)-log(x-4)-log(x-5)-log(x-6);
   	return z;
end
"""
	Ψ(x)
	computes the digamma of `x`
	# Examples
	```julia-repl
	julia> x = 1.0
	julia> Ψ(x)
	-0.5772156648761266
	```
"""
function Ψ(x::T) where T<: Real
	x=x+6.0
	p=1.0/abs2(x)
	p= (((0.004166666666667*p-0.003968253986254)*p+0.008333333333333)*p-0.083333333333333)*p
	p= p+log(x)-0.5/x-1.0/(x-1.0)-1.0/(x-2.0)-1.0/(x-3.0)-1.0/(x-4.0)-1.0/(x-5.0)-1.0/(x-6.0)
	return p
end
"""
	dΨ(x)
	computes the trigamma(derivaite of digamma) of `x`
	# Examples
	```julia-repl
	julia> x = 1.0
	julia> dΨ(x)
	1.6449340668506196
	```
"""
function dΨ(x::T) where T <: Real
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
"""
	dΨ(x)
	computes the shifted trigamma(derivaite of digamma) of `x`
"""
function dΨ_shifted(x::T) where T <: Real
	x = x + .5
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
"""
	δ(x,y)
	checks whether `x == y`
	# Examples
	```julia-repl
	julia> x = 1, y == 2
	julia> δ(x,y)
	false
	julia> x = 1, y == 1
	julia> δ(x,y)
	true
	```
"""
function δ(i::Int64,j::Int64)
	if i == j
		return 1
	else
		return 0
	end
end
"""
	dir_expectation(X)
	Dirichlet expectation of `log` of `X` vector
	# Examples
	```julia-repl
	julia> X = [1.0 ,2.0, 3.0, 4.0]
	julia> dir_expectation(X)
	4-element Array{Float64,1}:
	 -2.828968253942854
	 -1.828968253961494
	 -1.3289682539661583
	 -0.9956349206341892
	```
"""
function dir_expectation(X::T) where T <: AbstractVector{Float64}
    Ψ.(X) .- Ψ(sum(X))
end
"""
	dir_expectation!(X)
	in-place Dirichlet expectation of `log` of `X` vector into `Y` vector
	# Examples
	```julia-repl
	julia> X = [1.0 ,2.0, 3.0, 4.0];Y = zero(X)
	julia> dir_expectation!(Y,X)
	```
"""
function dir_expectation!(Y::T,X::T) where T <: AbstractVector{Float64}
    Y .= dir_expectation(X);
	return
end
"""
	dir_expectation2D(X)
	Dirichlet expectation of `log` of `X` matrix
	# Examples
	```julia-repl
	julia> X = [1.0 2.0; 3.0 4.0]
	julia> dir_expectation2D(X)
	2×2 Array{Float64,2}:
	 -2.82897  -1.82897
	 -1.32897  -0.995635
	```
"""

function dir_expectation2D(X::T) where T <: AbstractArray{Float64}
    Ψ.(X) .- Ψ(sum(X))
end
"""
	dir_expectation2D!(X)
	in-place Dirichlet expectation of `log` of `X` matrix into `Y` matrix
	# Examples
	```julia-repl
	julia> X = [1.0 2.0; 3.0 4.0]; Y = zero(X)
	julia> dir_expectation2D!(Y,X)
	```
"""
function dir_expectation2D!(Y::T,X::T) where T <: AbstractArray{Float64}
	Y .= dir_expectation2D(X);
	return
end

"""
	dir_expectationByRow!(X)
	in-place Dirichlet expectation of `log` of `X` matrix into `Y` matrix by row
	# Examples
	```julia-repl
	julia> X = [1.0 2.0; 3.0 4.0]; Y = zero(X)
	julia> dir_expectationByRow!(Y,X)
	```
"""
function dir_expectationByRow!(Y::T,X::T) where T <: AbstractArray{Float64}
	for (k,row) in enumerate(eachrow(X))
		Y[k,:] .= dir_expectation(row)
	end
	return
end
######################################################
######################################################
"""
	dir_expectation_shifted(X)
	shifted Dirichlet expectation of `log` of `X` vector
"""
function dir_expectation_shifted(X::T) where T <: AbstractVector{Float64}
    Ψ.(X .+ 0.5) .- Ψ(sum(X) + 0.5)
end
"""
	dir_expectation_shifted!(X)
	in-place shifted Dirichlet expectation of `log` of `X` vector into `Y` vector
"""
function dir_expectation_shifted!(Y::T,X::T) where T <: AbstractVector{Float64}
    Y .= dir_expectation_shifted(X);
	return
end
"""
	dir_expectation2D_shifted(X)
	shifted Dirichlet expectation of `log` of `X` matrix
"""
function dir_expectation2D_shifted(X::T) where T <: AbstractArray{Float64}
    Ψ.(X .+ 0.5) .- Ψ(sum(X) + 0.5)
end
"""
	dir_expectation2D_shifted!(X)
	in-place shifted Dirichlet expectation of `log` of `X` matrix into `Y` matrix
"""
function dir_expectation2D_shifted!(Y::T,X::T) where T <: AbstractArray{Float64}
	Y .= dir_expectation2D_shifted(X);
	return
end

"""
	dir_expectationByRow_shifted!(X)
	in-place shifted Dirichlet expectation of `log` of `X` matrix into `Y` matrix by row
"""
function dir_expectationByRow_shifted!(Y::T,X::T) where T <: AbstractArray{Float64}
	for (k,row) in enumerate(eachrow(X))
		Y[k,:] .= dir_expectation_shifted(row)
	end
	return
end

"""
    mean_dir_dot(matlist)

	computes the Dirichlet mean of a vector of matrices `matlist`
	# Examples
	```julia-repl
	julia> matlist = [[1.0 3.0; 2.0 4.0].+j for j in 0:1];matlist
	2-element Array{Array{Float64,2},1}:
	 [1.0 3.0; 2.0 4.0]
	 [2.0 4.0; 3.0 5.0]
	julia> mean_dir_dot(matlist)
	2-element Array{Array{Float64,2},1}:
	 [0.1 0.3; 0.2 0.4]
	 [0.14285714285714285 0.2857142857142857; 0.21428571428571427 0.35714285714285715]
	 ```
"""
function mean_dir_dot(gamma)
	theta_est = deepcopy(gamma)
	for i in 1:length(theta_est)
		s = sum(gamma[i])
		theta_est[i] ./= s
	end
	return theta_est
end
"""
    mean_dir_by_row(mat)

	computes the Dirichlet mean of rows of the matrix `mat`
	# Examples
	```julia-repl
	julia> mat = [1.0 3.0; 2.0 4.0];mat
	2×2 Array{Float64,2}:
	 1.0  2.0
	 3.0  4.0
	julia> vectorize_mat(mat)
	2×2 Array{Float64,2}:
	 0.25      0.75
	 0.333333  0.666667
	```
"""
function mean_dir_by_row(lambda_::Matrix{Float64})
	res = zeros(Float64, size(lambda_))
	for k in 1:size(lambda_, 1)
		res[k,:] .= mean(Dirichlet(lambda_[k,:]))
	end
	return res
end
"""
	sort_by_argmax!(X)
	sorts the rows of `X` matrix based on the values in their columns and \n
	returns the updated matrix and the row indices
	# Examples
	```julia-repl
	julia> X = [1.0 2.0; 3.0 2.0; 2.0 4.0]; X
	 1.0  2.0
	 3.0  2.0
	 2.0  4.0
	3×2 Array{Float64,2}:
	 3.0  2.0
	 2.0  3.0
	 1.0  3.0
	julia> X,inds = sort_by_argmax!(X);
	julia> X
	3.0  2.0
	1.0  2.0
	2.0  4.0
	julia> inds
	2
	1
	3
	```
"""
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
"""
	find_all!(val, X)
	return a `Bool` vector of places whether `val` occured or not in `X`
	# Examples
	```julia-repl
	julia> val = 2; X = [2, 3, 4, 5, 2];
	julia> find_all(val, X)
	2-element Array{Int64,1}:
	 1
	 5
	```
"""
function find_all(val::Int64, doc::Vector{Int64})
	findall(x -> x == val, doc)
end
"""
	get_ρ!(i, settings)
	computes the learning rate according to \n
	(S+`i`)^-κ
"""
function get_ρ(i::Int64, settings::Settings)
	return (settings._LR_OFFSET+convert(Float64, i))^(-settings._LR_KAPPA)
end
"""
	get_ρ!(epoch, mb,mb_index, settings)
	computes the learning rate according to \n
	(S+`epoch` + (`mb_index`/|`mb`|))^-κ
"""
function get_ρ(epoch_count, mb, mindex,settings::Settings)
	return (settings._LR_OFFSET+epoch_count+(mindex/length(mb)))^(-settings._LR_KAPPA)
end
"""
	mean_change(new_, old_)
	computes the mean change of two same-shape arrays `new_` and `old_`
	# Examples
	```julia-repl
	julia>  new_ = [1.1, 1.2, 1.3, 1.4];old_ = [1.0, 1.0, 1.0, 1.0];
	julia> mean_change(new_, old_)
	0.25
	```
"""
function mean_change(new::AbstractArray{R}, old::AbstractArray{R}) where  {R<:AbstractFloat}
	n = length(new)
	change = sum(abs.(new .- old))/n
	return(change)
end
################################################################################
##############################  Data Wrangling  ################################
################################################################################
"""
	fix_corp!(model, folder)
	creates documents within the corpuses with unique terms for both views \n
		and saves it to `folder`
	```
"""
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
"""
	figure_sparsity!(model, sparsity, all_, folder)
	modifies corpuses according to the two arguments `all_` and `sparsity` \n
	and saves them to `folder`
	if `all_=true`, `sparsity` proportion of the words within documents are
		removed and corpuses and documents are adjusted accordingly.
	if `all_=false`, `sparsity` proportion of the documents are
		removed and corpuses are adjusted accordingly.
	```
"""
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
					corp._docs[j]._terms = Int64[]
					corp._docs[j]._counts = [0]
					corp._docs[j]._length = sum(corp._docs[j]._counts)
				else
					y2 = unique(vcat(y2, corp._docs[j]._terms))
				end
			end
			if length(unique(y2)) == corp._V
				model._corpus2 = deepcopy(corp)
				break
			else
				y2 = Int64[]
			end
		end
	end
	corp2_sparse = deepcopy(model._corpus2)
	@save "$(folder)/corpus2_sparse" corp2_sparse
end
"""
	setup_hmap(model, h,D)
	returns a `Bool` vector `h_map` where `sum(h_map)` = `floor(h×D)`\n
	the functions makes sure that by setting aside a heldout sample\n
	vocabulary of the train does not become any smaller.
"""
function setup_hmap(folder::String,model::MVD, h::Float64,_D::Int64)
	h_count = convert(Int64, floor(h*_D))
	corp1 = deepcopy(model._corpus1)
	corp2 = deepcopy(model._corpus2)

	while true
		h_map = repeat([false], _D)
		inds = sample(1:_D, h_count, replace=false, ordered=true)
		h_map[inds] .= true
		x1 = [corp1._docs[i]._terms for i in collect(1:corp1._D)[.!h_map]]
		x2 = [corp2._docs[i]._terms for i in collect(1:corp2._D)[.!h_map]]
		cond1 = any(.!isempty.(x1)) &&  any(.!isempty.(x2))
		cond2 = (length(unique(vcat(x1...))) == corp1._V) && (length(unique(vcat(x2...))) == corp2._V)
		if cond1 & cond2
			@save "$(folder)/h_map" h_map
			return h_map
		end
	end
end

function preprocess!(folder::String, model::MVD, sparsity::Float64, all_::Bool, h, _D, mb_size)
	fix_corp!(model, folder)
	figure_sparsity!(model,sparsity,all_, folder)
	args_names, args = create_test(folder, model, h, _D, mb_size)
	return args_names, args
end
"""
split_ho_obs(model, h_map)

	for each of the two views, given the `h_map` creates test cases,
	where parts of the documents in each mode are observed and some
	that are not. It makes sure that unobserved ones do not include
	any words that are not previously accounted for in the vocabulary.
	currently the split is 90% observed 10% unobserved.
"""
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

function create_test(folder, model, h, _D, mb_size)
	h_map = setup_hmap(folder, model,h,_D)
	mbs, nb = epoch_batches(collect(1:_D)[.!h_map], mb_size)
	mindex, epoch_count = 1,0
	args_names = [:h_map,:mbs, :nb, :mindex, :epoch_count,:hos1_dict,:obs1_dict,:hos2_dict,:obs2_dict]
	args = h_map, mbs, nb, mindex, epoch_count, split_ho_obs(model, h_map)...
	return (args_names, args)
end

"""
	epoch_batches(ids_, size_)
	given a list of `ids_` creates a set of non-overlapping \n
	minibatches of `size_` and their length
	# Examples
	```julia-repl
	julia> ids_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; size_ = 2;
	julia> mb, nb = epoch_batches(ids_, size_);
	julia> mb
	6-element Array{Array{Int64,1},1}:
	[7, 8]
	[2, 4]
	[9, 3]
	[5, 10]
	[1, 6]
	[4]
	julia> nb
	6
	```
"""
function epoch_batches(_train_ids::Vector{Int64}, mb_size::Int64)
	N_ = length(_train_ids)
	div_ = div(N_, mb_size)
	nb = (div_ * mb_size - N_) < 0 ? div_ + 1 : div_
	y = shuffle(_train_ids)
	x = [Int64[] for _ in 1:nb]
	for n in 1:nb
		while length(x[n]) < mb_size && !isempty(y)
			push!(x[n],pop!(y))
		end
	end
	return x, nb
end
#####################MACROS#################
############################################
macro shift(var)
	return :($var .+ 0.5)
end
macro bump(var)
	return :($var .+ 1e-100)
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
print("")
