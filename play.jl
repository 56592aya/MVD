using Plots, FileIO, JLD2
sc = joinpath(homedir(), "Dropbox/Arash/EUR/Workspace/MVD/")

if !isdir("pics")
	mkdir("pics")
	Plots.plot([1,2,3,4], [1,2,3,4])
	include(joinpath(sc,"loader.jl"))
	cd = pwd()
	@load "../truth" Truth_Params
else
	include(joinpath(sc,"loader.jl"))
	pwd
	@load "../truth" Truth_Params
end
####


include(joinpath(sc,"assess.jl"))
model = read_model(eee);
theta_est,ϕ1_est, ϕ2_est, ϕ1_truth, ϕ2_truth,theta_truth, inds1, inds2 = do_ϕs(model);
theta_truth_1,theta_truth_2,theta_est_1,theta_est_2 = do_plots(model, theta_est,ϕ1_est, ϕ2_est, ϕ1_truth, ϕ2_truth, theta_truth,inds1, inds2);
@load "h_map" h_map;
