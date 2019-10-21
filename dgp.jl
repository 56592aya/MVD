struct Params
	N::Int64
	K1::Int64
	K2::Int64
	V1::Int64
	V2::Int64
	Α_vec::Vector{Float64}
	Α::Matrix{Float64}
	Θ_vec::Matrix{Float64}
	Θ::Vector{Matrix{Float64}}
	β1::Matrix{Float64}
	β2::Matrix{Float64}
	Β1::Matrix{Float64}
	Β2::Matrix{Float64}
end



function create_Alpha(K1::Int64, K2::Int64, prior)
	tot_dim = K1*K2
	res = rand(Dirichlet(tot_dim, prior))
	Res = reshape(res, (K2, K1))
	Res = permutedims(Res, (2,1))
    return res, Res
end
function create_Alpha_manual(K1::Int64, K2::Int64, prior)

	@assert K1 == K2
	Res = zeros(Float64, (K1, K2))
	for i in 1:K1
		Res[i,i] = prior
		for j in 1:K1
			if j  == i
				continue;
			else
				Res[i,j] = rand(Distributions.Uniform(0.0, .1))
			end
		end
	end
	Res ./ sum(Res)
	res = vectorize_mat(Res)
    return res, Res
end

function create_Theta(vec::Vector{Float64}, N::Int64, K1::Int64, K2::Int64)
	res = rand(Distributions.Dirichlet(vec),N)
	Res = [permutedims(reshape(res[:,i], (K2,K1)), (2,1)) for i in 1:N]
    return permutedims(res, (2, 1)), Res
end


function create_B(beta_prior::Matrix{Float64}, K::Int64, V::Int64)
	B = zeros(Float64, (K, V))
	for k in 1:K
		B[k,:] = rand(Dirichlet(beta_prior[k,:]))
	end
    return B
end

function create_doc(wlen::Int64, topic_dist_vec::Vector{Float64},
	                term_topic_dist::Matrix{Float64}, mode_::Int64,
					K1::Int64, K2::Int64)
	doc = Int64[]
	for w in 1:wlen

		topic_temp = rand(Distributions.Categorical(topic_dist_vec))
		x = zeros(Int64, K1*K2)
		x[topic_temp] = 1
		X = matricize_vec(x, K1, K2)
		where_ = findall(x -> x == 1, X)[1]
		row, col = where_.I
		topic = mode_ == 1 ? row : col
		term = rand(Distributions.Categorical(term_topic_dist[topic,:]))
		doc = vcat(doc, term)
	end
	return doc
end
function create_corpux(N::Int64, vec_list::Matrix{Float64}, B::Matrix{Float64},
	 				   K1::Int64, K2::Int64, wlens::Vector{Int64}, mode_::Int64)

	corpus = [Int64[] for i in 1:N]
	for i in 1:N

		doc  = create_doc(wlens[i], vec_list[i,:] ,B, mode_, K1, K2)
		corpus[i] = vcat(corpus[i], doc)
	end
	return corpus
end

function Create_Truth(N, K1, K2, V1, V2, prior,β1_single, β2_single, wlen1_single, wlen2_single, manual)

	α, Α = manual ? create_Alpha_manual(K1, K2,prior) : create_Alpha(K1, K2,prior)
	θ,Θ = create_Theta(α, N, K1, K2)
	β1 = ones(Float64, (K1, V1)) .* β1_single
	Β1 = create_B(β1, K1, V1)
	β2 = ones(Float64, (K2, V2)) .* β2_single
	Β2 = create_B(β2, K2, V2)
	wlens1 = [wlen1_single for i in 1:N]
	wlens2 = [wlen2_single for i in 1:N]
	corp1 = create_corpux(N, θ, Β1,K1,K2, wlens1, 1)
	corp2 = create_corpux(N, θ, Β2,K1,K2, wlens2, 2)
	return α,Α, θ,Θ, Β1, Β2, β1, β2, V1, V2, corp1, corp2
end


function simulate_data(N, K1, K2, V1, V2,prior,β1_single_truth, β2_single_truth,wlen1_single, wlen2_single, manual)
	y1 = Int64[]
 	y2 = Int64[]
 	while true
		# prior,β1_single_truth,β2_single_truth = .99, .2, .2
		α_truth,Α_truth, θ_truth,Θ_truth,
 		Β1_truth, Β2_truth, β1_truth, β2_truth,V1, V2, corp1, corp2 =
 		Create_Truth(N, K1, K2, V1, V2, prior,β1_single_truth, β2_single_truth, wlen1_single, wlen2_single, manual)
		for i in 1:N
			# global y1, y2
         	y1 = unique(y1)
 		  	y2 = unique(y2)
 		    y1 = vcat(y1, corp1[i])
 		    y2 = vcat(y2, corp2[i])
 		end
 		y1 = unique(y1)
 		y2 = unique(y2)
 		if ((length(y1) == V1) && (length(y2) == V2))
             return α_truth,Α_truth, θ_truth,Θ_truth,Β1_truth, Β2_truth, β1_truth, β2_truth,V1, V2, corp1, corp2
 		else
         	y1 = Int64[]
         	y2 = Int64[]
		end
	 end
end
