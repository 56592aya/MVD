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
theta_est,B1_est, B2_est, B1_truth, B2_truth,theta_truth, inds1, inds2 = do_Bs(model);
theta_truth_1,theta_truth_2,theta_est_1,theta_est_2 = do_plots(model, theta_est,B1_est, B2_est, B1_truth, B2_truth, theta_truth,inds1, inds2);
@load "h_map" h_map
(mean(model.γ[.!h_map])/25)[inds1, inds2]
 update_alpha_newton!(model, count_params, h_map)[inds1, inds2]
