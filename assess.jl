using Plots, FileIO, JLD2

if !isdir("pics")
	mkdir("pics")
	Plots.plot([1,2,3,4], [1,2,3,4])
	include("../../loader.jl")
	@load "../truth" Truth_Params
else
	include("../../loader.jl")
	@load "../truth" Truth_Params
end
####

@load "model_at_epoch_$(eee)" model

B1_est = estimate_B(model.b1);
B2_est = estimate_B(model.b2);
theta_est = estimate_thetas(model.γ);
B1_truth = deepcopy(Truth_Params.Β1);
B2_truth = deepcopy(Truth_Params.Β2);
theta_truth = deepcopy(Truth_Params.Θ);

l = collect(1:size(B1_truth,1));
m = 1000000.0;
for i in 1:length(l)
	global m
	for j in 1:length(l)
		val = sqrt(sum( (B1_truth[i,:] .- B1_est[j,:]).^2))
		if val < m
			m = val
			l[i] = j
		end
	end
	m=1000000.0
end
inds1 = deepcopy(l);
l = collect(1:size(B2_truth,1));
m = 1000000.0;
for i in 1:length(l)
	global m
	for j in 1:length(l)
		val = sqrt(sum( (B2_truth[i,:] .- B2_est[j,:]).^2))
		if val < m
			m = val
			l[i] = j
		end
	end
	m=1000000.0
end
inds2 = deepcopy(l);
println(inds1);
println(inds2);
if length(unique(inds1)) == length(inds1) && length(unique(inds2)) == length(inds2)
	p1b = Plots.heatmap(B1_truth, yflip=true)
	p1a = Plots.heatmap(B1_est, yflip=true)
	p2b = Plots.heatmap(B2_truth, yflip=true)
	p2a = Plots.heatmap(B2_est, yflip=true)
	p1a = Plots.heatmap(B1_est[inds1,:], yflip=true)
	p2a = Plots.heatmap(B2_est[inds2,:], yflip=true)
	plot(p1a, p2a, p1b, p2b, layout =(2, 2), legend=false)
	savefig("pics/betas.png")





	theta_truth_1 = zeros(Float64, (length(theta_truth), size(B1_truth,1)));
	for i in 1:size(theta_truth_1,1)
		for j in 1:length(inds1)
			theta_truth_1[i,j] = sum(theta_truth[i][j,:])
		end
	end
	theta_truth_2 = zeros(Float64, (length(theta_truth), size(B2_truth,1)));
	for i in 1:size(theta_truth_2,1)
		for j in 1:length(inds2)
			theta_truth_2[i,j] = sum(theta_truth[i][:,j])
		end
	end

	theta_est_1 = zeros(Float64, (length(theta_truth), size(B1_truth,1)));
	theta_est_2 = zeros(Float64, (length(theta_truth), size(B2_truth,1)));

	for i in 1:size(theta_est_1,1)
		for j in 1:size(B1_truth,1)
			theta_est_1[i,j] = sum(theta_est[i][inds1[j],:])
		end
	end
	for i in 1:size(theta_est_2,1)
		for j in 1:size(B2_truth,1)
			theta_est_2[i,j] = sum(theta_est[i][:,inds2[j]])
		end
	end
	x = collect(range(0.0, 1.0, length=100));
	y = collect(range(0.0, 1.0, length=100));
	@load "h_map" h_map
	plts = [];
	for k in 1:size(B1_est[inds1,:],1)
		global plts
		p = scatter(theta_truth_1[:,k], theta_est_1[:,k], grid=false, aspect_ratio=:equal,legend=false);plot!(x, y, linewidth=3);
		plts = vcat(plts, p)
	end
	Plots.plot(plts..., layout =(1, size(B1_est,1)), legend=false)
	savefig("pics/thetas1.png")

	plts = [];
	for k in 1:length(inds1)
		global plts
		p = scatter(theta_truth_1[.!(h_map),k], theta_est_1[.!(h_map),k], grid=false, aspect_ratio=:equal,legend=false);plot!(x, y, linewidth=3);
		plts = vcat(plts, p)
	end
	Plots.plot(plts..., layout =(1, length(inds1)), legend=false)
	savefig("pics/thetas1_train.png")
	plts = [];
	for k in 1:length(inds1)
		global plts
		p = scatter(theta_truth_1[h_map,k], theta_est_1[h_map,k], grid=false, aspect_ratio=:equal,legend=false);plot!(x, y, linewidth=3);
		plts = vcat(plts, p)
	end
	Plots.plot(plts..., layout =(1, length(inds1)), legend=false)
	savefig("pics/thetas1_ho.png")



	plts = [];
	for k in 1:length(inds2)
		global plts
		p = scatter(theta_truth_2[:,k], theta_est_2[:,k], grid=false, aspect_ratio=:equal,legend=false);plot!(x, y, linewidth=3);
		plts = vcat(plts, p)
	end
	Plots.plot(plts..., layout =(1, length(inds2)), legend=false)
	savefig("pics/thetas2.png")

	plts = [];
	for k in 1:length(inds2)
		global plts
		p = scatter(theta_truth_2[.!(h_map),k], theta_est_2[.!(h_map),k], grid=false, aspect_ratio=:equal,legend=false);plot!(x, y, linewidth=3);
		plts = vcat(plts, p)
	end
	Plots.plot(plts..., layout =(1, length(inds2)), legend=false)
	savefig("pics/thetas2_train.png")
	plts = [];
	for k in 1:length(inds2)
		global plts
		p = scatter(theta_truth_2[h_map,k], theta_est_2[h_map,k], grid=false, aspect_ratio=:equal,legend=false);plot!(x, y, linewidth=3);
		plts = vcat(plts, p)
	end
	Plots.plot(plts..., layout =(1, length(inds2)), legend=false)
	savefig("pics/thetas2_ho.png")

	absent_map = [i for i in 1:length(model.Corpus1.docs) if model.Corpus2.docs[i].len  == 0]
	if !isempty(absent_map)
		present_map = [i for i in 1:length(model.Corpus1.docs) if model.Corpus2.docs[i].len != 0]
		plts = [];
		for k in 1:length(inds2)
			global plts
			p = scatter(theta_truth_2[present_map,k], theta_est_2[present_map,k], grid=false, aspect_ratio=:equal,legend=false);plot!(x, y, linewidth=3);
			plts = vcat(plts, p)
		end
		Plots.plot(plts..., layout =(1, length(inds2)), legend=false)
		savefig("pics/thetas2_present.png")
		plts = [];
		for k in 1:length(inds2)
			global plts
			p = scatter(theta_truth_2[absent_map,k], theta_est_2[absent_map,k], grid=false, aspect_ratio=:equal,legend=false);plot!(x, y, linewidth=3);
			plts = vcat(plts, p)
		end
		Plots.plot(plts..., layout =(1, length(inds2)), legend=false)
		savefig("pics/thetas2_absent.png")
	end
	p1 = Plots.heatmap(model.Alpha[inds1, inds2], yflip = true)
	savefig("pics/model_Alpha.png")
	p2 = Plots.heatmap(Truth_Params.Α, yflip = true)
	savefig("pics/true_Alpha.png")

end
#
#
# ####### IMPUTE ########
# function impute_theta(model, i)
# 	doc = Int64[]
# 	x = rand(Distributions.Dirichlet(vectorize_mat(model.Alpha)))
# 	Res = permutedims(reshape(x[:,i], (model.K2,model.K1)), (2,1))
#
# 	for w in 1:200
#
# 		topic_temp = rand(Distributions.Categorical(vectorize_mat(Res)))
# 		x = zeros(Int64, model.K1*model.K2)
# 		x[topic_temp] = 1
# 		X = matricize_vec(x, model.K1, model.K2)
# 		where_ = findall(x -> x == 1, X)[1]
# 		row, col = where_.I
# 		# row = Int64(ceil(topic_temp/K2))
# 		# col = topic_temp - (row-1)*K2
# 		topic =  col
# 		term = rand(Distributions.Categorical(B2_est[topic,:]))
# 		doc = vcat(doc, term)
# 	end
# 	return doc
# end
# doc = impute_theta(model, i)
# zeroer_i = zeros(Float64, (model.K1, model.K2))
# zeroer_mb_1 = zeros(Float64, (model.K1,model.Corpus1.V))
# zeroer_mb_2 = zeros(Float64, (model.K2,model.Corpus2.V))
#
# doc = model.Corpus1.docs[1].terms
# uniqs1 = unique(doc)
# counts1 = Int64[]
# for u in uniqs1
# 	counts1 = vcat(counts1, length(find_all(u, doc)))
# end
# c1.docs[i] = Document(uniqs1,counts1,doc1.len)
 # x = (sum(model.γ[.!(h_map)]) - sum(model.alpha_sstat[.!(h_map)]))/sum(.!(h_map))
