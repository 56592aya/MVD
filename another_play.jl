absent_map = [i for i in 1:length(model.Corpus1.docs) if model.Corpus2.docs[i].len  == 0]

x = mean(model.γ[.!h_map])
x .-= .5*model.K1
x[x.< 0.0 ] .= 1e-10
x ./=sum(x)
model.α = deepcopy(x) .+ .5


for i in absent_map
    model.γ[i] = rand(Gamma(100.0, 0.01), (model.K1,model.K2))
end
zeroer_i = zeros(Float64, (model.K1, model.K2)  )
zeroer_mb_1 = zeros(Float64, (model.K1, model.Corpus1.V))
zeroer_mb_2 = zeros(Float64, (model.K2,model.Corpus2.V))
copyto!(model.sum_π_1_mb, zeroer_mb_1)
copyto!(model.sum_π_2_mb, zeroer_mb_2)
copyto!(model.sum_π_1_i,  zeroer_i)
copyto!(model.sum_π_2_i, zeroer_i)


for i in absent_map
    update_ElogΘ_i!(model, i)
    doc1 = model.Corpus1.docs[i]
    doc2 = model.Corpus2.docs[i]
    copyto!(model.old_γ, model.γ[i])
    gamma_flag = false

#######
    for _ in 1:2000
        model.sum_π_1_i .= zeroer_i
        model.sum_π_2_i .= zeroer_i
        for (w,val) in enumerate(doc1.terms)
            optimize_π_iw!(model, i,1,val)
            @. model.sstat_i = doc1.counts[w] * model.π_temp
            @.(model.sum_π_1_i += model.sstat_i)
        end
        for (w,val) in enumerate(doc2.terms)
            optimize_π_iw!(model, i,2,val)
            @. model.sstat_i = doc2.counts[w] * model.π_temp
            @.(model.sum_π_2_i += model.sstat_i)
        end
        optimize_γi_perp!(model, i)
        update_ElogΘ_i!(model,i)
        gamma_change = mean_change(model.γ[i], model.old_γ)
        println(gamma_change)
        model.old_γ .= model.γ[i]
    end
end




theta_est,ϕ1_est, ϕ2_est, ϕ1_truth, ϕ2_truth,theta_truth, inds1, inds2 = do_ϕs(model);
theta_truth_1,theta_truth_2,theta_est_1,theta_est_2 = do_plots(model, theta_est,ϕ1_est, ϕ2_est, ϕ1_truth, ϕ2_truth, theta_truth,inds1, inds2);


############3
