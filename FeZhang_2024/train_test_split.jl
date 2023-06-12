using ACE1pack, JuLIP, Random 


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


##


data_file = @__DIR__() * "/Fe_data.xyz"
data = read_extxyz(data_file)
training_data, testing_data = train_test_split(data, 0.1)
@assert length(data) == length(training_data) + length(testing_data)
@assert isempty(setdiff(setdiff(data, training_data), testing_data))

JuLIP.write_extxyz(@__DIR__() * "/Fe_train.xyz", training_data)
JuLIP.write_extxyz(@__DIR__() * "/Fe_test.xyz", testing_data)