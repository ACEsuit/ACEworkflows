
function train_test_split(data, test_pct = 0.2)
   cfgtypes = unique([ get_data(at,"config_type") for at in data])
   datacfg = Dict{String,Vector{Atoms{Float64}}}()
   for cfgt in cfgtypes
       datacfg[cfgt] = Atoms{Float64}[]
   end
   for at in data
       push!(datacfg[get_data(at,"config_type")], at)
   end
   train_data = Atoms{Float64}[]
   test_data = Atoms{Float64}[]
   for cfgt in cfgtypes
       Ncfg = length(datacfg[cfgt])
       Ntest = round(Int, Ncfg * test_pct)
       ind_test = sort(shuffle(1:Ncfg)[1:Ntest])
       ind_train = setdiff(1:Ncfg, ind_test)
       append!(train_data, datacfg[cfgt][ind_train])
       append!(test_data, datacfg[cfgt][ind_test])
   end   
   return train_data, test_data 
end