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
    for _ in 1:1000
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
        #println(gamma_change)
        model.old_γ .= model.γ[i]
    end
end






############3
