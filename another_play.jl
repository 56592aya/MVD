model.γ[i] = rand(Gamma(100.0, 0.01), (model.K1,model.K2))

copyto!(model.sum_phi_1_mb, zeroer_mb_1)
copyto!(model.sum_phi_2_mb, zeroer_mb_2)
copyto!(model.sum_phi_1_i,  zeroer_i)
copyto!(model.sum_phi_2_i, zeroer_i)



update_Elogtheta_i!(model, i)
doc1 = model.Corpus1.docs[i]
doc2 = model.Corpus2.docs[i]
copyto!(model.old_γ, model.γ[i])
gamma_c = false

#######
model.sum_phi_1_i .= zeroer_i
model.sum_phi_2_i .= zeroer_i
for (w,val) in enumerate(doc1.terms)
    optimize_phi_iw!(model, i,1,val)
    @. model.sstat_i = doc1.counts[w] * model.temp
    @.(model.sum_phi_1_i += model.sstat_i)
end

optimize_γi_perp!(model, i)
update_Elogtheta_i!(model,i)
gamma_change = mean_change(model.γ[i], model.old_γ)
println(gamma_change)
model.old_γ .= model.γ[i]
