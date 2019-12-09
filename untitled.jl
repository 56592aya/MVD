MAXITER = 90000
S, κ = 256.0, .6
every = 1
VI_CONVERGED = false;  MAX_VI_ITER = MAXITER;  MAX_ALPHA_ITER = 1000;  MAX_GAMMA_ITER = 1000;
MAX_ALPHA_DECAY= 10;  ALPHA_DECAY_FACTOR = .8;  ALPHA_THRESHOLD = 1e-5;  GAMMA_THRESHOLD =1e-3
VI_THRESHOLD = 1e-8;  EVAL_EVERY = every; LR_OFFSET, LR_KAPPA = S, κ;

D2 = sum([1 for i in collect(1:model._corpus1._D)[.!h_map] if model._corpus2._docs[i]._length != 0])
count_params = TrainCounts(model._corpus1._D-sum(h_map),D2, model._K1, model._K2)
dir_expectationByRow_shifted!(model._elogϕ1, model._λ1)     # <=====  HERE
dir_expectationByRow_shifted!(model._elogϕ2, model._λ2)    # <=====  HERE

settings = Settings(model._K1, model._K2, model._corpus1, model._corpus2,
                    MAX_VI_ITER,MAX_ALPHA_ITER,MAX_GAMMA_ITER,MAX_ALPHA_DECAY,
                    ALPHA_DECAY_FACTOR,ALPHA_THRESHOLD,GAMMA_THRESHOLD,VI_THRESHOLD,
                    EVAL_EVERY, LR_OFFSET, LR_KAPPA)

_C1 = model._corpus1._docs;	_C2 = model._corpus2._docs;	_V1 = model._corpus1._V; _V2 = model._corpus2._V;
_D = model._corpus1._D;	_D1 = count_params._D1;	_D2 = count_params._D2;	_K1 = count_params._K1;
_K2 = count_params._K2;	_terms1 = [_C1[i]._terms for i in 1:model._corpus1._D];
_terms2 = [_C2[i]._terms for i in 1:model._corpus2._D];_counts1 = [_C1[i]._counts for i in 1:model._corpus1._D];
_counts2 = [_C2[i]._counts for i in 1:model._corpus2._D];_lengths1 = [_C1[i]._length for i in 1:model._corpus1._D];
_lengths2 = [_C2[i]._length for i in 1:model._corpus2._D];	_train_ids = collect(1:_D)[.!h_map]
perp1_list = Float64[];	perp2_list = Float64[];burnin = true

js = [j for j in collect(1:10000)[.!h_map] if model._corpus2._docs[j]._length == 0]
mb = js
 _d1 = length(mb);_d2 = sum([_lengths2[d] != 0  for d in mb]);
 init_γs!(model, mb)
 args = init_sstats!(model, settings, mb)
 update_local!(model, settings, mb,_terms1,_counts1,_terms2,_counts2, args...)

 theta_est = mean_dir_dot(model._γ[mb]);
 theta_truth = deepcopy(Truth_Params.Θ);

 theta_truth_1 = zeros(Float64, (length(theta_truth), size(ϕ1_truth,1)));
 for i in 1:size(theta_truth_1,1)
     for j in 1:length(inds1)
         theta_truth_1[i,j] = sum(theta_truth[i][j,:])
     end
 end
 theta_truth_2 = zeros(Float64, (length(theta_truth), size(ϕ2_truth,1)));
 for i in 1:size(theta_truth_2,1)
     for j in 1:length(inds2)
         theta_truth_2[i,j] = sum(theta_truth[i][:,j])
     end
 end

 theta_est_1 = zeros(Float64, (length(js), size(ϕ1_truth,1)));
 theta_est_2 = zeros(Float64, (length(js), size(ϕ2_truth,1)));
 for i in 1:size(theta_est_1,1)
     for j in 1:size(ϕ1_truth,1)
         theta_est_1[i,j] = sum(theta_est[i][inds1[j],:])
     end
 end
 for i in 1:size(theta_est_2,1)
     for j in 1:size(ϕ2_truth,1)
         theta_est_2[i,j] = sum(theta_est[i][:,inds2[j]])
     end
 end

 k = 1
 Plots.scatter(theta_truth_2[js,k], theta_est_2[:,k], grid=false, aspect_ratio=:equal,legend=false)
