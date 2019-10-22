function read_model(eee)
	@load "model_at_epoch_$(eee)" model
	return model
end
function do_Bs(model)
	B1_est = estimate_B(model.b1);
	B2_est = estimate_B(model.b2);
	theta_est = estimate_thetas(model.γ);
	B1_truth = deepcopy(Truth_Params.Β1);
	B2_truth = deepcopy(Truth_Params.Β2);
	theta_truth = deepcopy(Truth_Params.Θ);

	l = collect(1:size(B1_truth,1));
	m = 1000000.0;
	for i in 1:length(l)
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
	return theta_est,B1_est, B2_est, B1_truth, B2_truth,theta_truth, inds1, inds2
end

function do_plots(model,theta_est, B1_est, B2_est, B1_truth, B2_truth, theta_truth,inds1, inds2)

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
			p = scatter(theta_truth_1[:,k], theta_est_1[:,k], grid=false, aspect_ratio=:equal,legend=false);plot!(x, y, linewidth=3);
			plts = vcat(plts, p)
		end
		Plots.plot(plts..., layout =(1, size(B1_est,1)), legend=false)
		savefig("pics/thetas1.png")

		plts = [];
		for k in 1:length(inds1)
			p = scatter(theta_truth_1[.!(h_map),k], theta_est_1[.!(h_map),k], grid=false, aspect_ratio=:equal,legend=false);plot!(x, y, linewidth=3);
			plts = vcat(plts, p)
		end
		Plots.plot(plts..., layout =(1, length(inds1)), legend=false)
		savefig("pics/thetas1_train.png")
		plts = [];
		for k in 1:length(inds1)
			p = scatter(theta_truth_1[h_map,k], theta_est_1[h_map,k], grid=false, aspect_ratio=:equal,legend=false);plot!(x, y, linewidth=3);
			plts = vcat(plts, p)
		end
		Plots.plot(plts..., layout =(1, length(inds1)), legend=false)
		savefig("pics/thetas1_ho.png")



		plts = [];
		for k in 1:length(inds2)
			p = scatter(theta_truth_2[:,k], theta_est_2[:,k], grid=false, aspect_ratio=:equal,legend=false);plot!(x, y, linewidth=3);
			plts = vcat(plts, p)
		end
		Plots.plot(plts..., layout =(1, length(inds2)), legend=false)
		savefig("pics/thetas2.png")

		plts = [];
		for k in 1:length(inds2)
			p = scatter(theta_truth_2[.!(h_map),k], theta_est_2[.!(h_map),k], grid=false, aspect_ratio=:equal,legend=false);plot!(x, y, linewidth=3);
			plts = vcat(plts, p)
		end
		Plots.plot(plts..., layout =(1, length(inds2)), legend=false)
		savefig("pics/thetas2_train.png")
		plts = [];
		for k in 1:length(inds2)
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
				p = scatter(theta_truth_2[present_map,k], theta_est_2[present_map,k], grid=false, aspect_ratio=:equal,legend=false);plot!(x, y, linewidth=3);
				plts = vcat(plts, p)
			end
			Plots.plot(plts..., layout =(1, length(inds2)), legend=false)
			savefig("pics/thetas2_present.png")
			plts = [];
			for k in 1:length(inds2)
				p = scatter(theta_truth_2[absent_map,k], theta_est_2[absent_map,k], grid=false, aspect_ratio=:equal,legend=false);plot!(x, y, linewidth=3);
				plts = vcat(plts, p)
			end
			Plots.plot(plts..., layout =(1, length(inds2)), legend=false)
			savefig("pics/thetas2_absent.png")
		end
		p1 = Plots.heatmap(model.Alpha[inds1, inds2])
		savefig("pics/model_Alpha.png")
		p2 = Plots.heatmap(Truth_Params.Α)
		savefig("pics/true_Alpha.png")
		return theta_truth_1,theta_truth_2,theta_est_1,theta_est_2
	else
		return nothing, nothing, nothing, nothing
	end
end
