# function init_params(K1::Int64,K2::Int64, beta1_prior_, beta2_prior_,
# 	 alpha_prior_, corpus1_, corpus2_)
# 	N = max(corpus1_.N, corpus2_.N)
# 	alpha_vec = rand(Uniform(0.0,alpha_prior_), (K1*K2))
# 	Alpha = matricize_vec(alpha_vec, K1, K2)
# 	B1 = collect(repeat(rand(Uniform(0.0, beta1_prior_), corpus1_.V), inner=(1, K1))')
# 	B2 = collect(repeat(rand(Uniform(0.0, beta2_prior_), corpus2_.V), inner=(1, K2))')
# 	γ = [ones(Float64, (K1, K2)) for i in 1:N]
# 	b1 = deepcopy(B1)
# 	b2 = deepcopy(B2)
# 	Elog_B1 = zeros(Float64, (K1, corpus1_.V))
# 	Elog_B2 = zeros(Float64, (K2, corpus2_.V))
# 	Elog_Theta = [zeros(Float64, (K1, K2)) for i in 1:N]
# 	zeroer_i = zeros(Float64, (K1, K2))
# 	zeroer_mb_1 = zeros(Float64, (K1,corpus1_.V))
# 	zeroer_mb_2 = zeros(Float64, (K2,corpus2_.V))
# 	sum_phi_1_i = similar(zeroer_i)
# 	sum_phi_2_i = similar(zeroer_i)
# 	sum_phi_1_mb = similar(zeroer_mb_1)
# 	sum_phi_2_mb = similar(zeroer_mb_2)
# 	old_γ = similar(zeroer_i)
# 	old_b1 = similar(b1)
# 	old_b2 = similar(b2)
# 	old_Alpha = similar(Alpha)
# 	old_B1 = similar(B1)
# 	old_B2 = similar(B2)
# 	temp = similar(zeroer_i)
# 	sstat_i = similar(zeroer_i)
# 	sstat_mb_1 = zeros(Float64, K1)
# 	sstat_mb_2 = zeros(Float64, K2)
# 	#alpha_sstat = [deepcopy(zeroer_i) for i in 1:N]
# 	return 	Alpha,old_Alpha,B1,old_B1,B2,old_B2,Elog_B1,Elog_B2,Elog_Theta,γ,old_γ,b1,old_b1,b2,old_b2,
# 	temp,sstat_i,sstat_mb_1,sstat_mb_2,sum_phi_1_mb,sum_phi_2_mb,sum_phi_1_i,sum_phi_2_i#, alpha_sstat
# end
