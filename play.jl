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

include("../../assess.jl")
model = read_model(eee);
theta_est,ϕ1_est, ϕ2_est, ϕ1_truth, ϕ2_truth,theta_truth, inds1, inds2 = do_ϕs(model);
theta_truth_1,theta_truth_2,theta_est_1,theta_est_2 = do_plots(model, theta_est,ϕ1_est, ϕ2_est, ϕ1_truth, ϕ2_truth, theta_truth,inds1, inds2);
@load "h_map" h_map;
