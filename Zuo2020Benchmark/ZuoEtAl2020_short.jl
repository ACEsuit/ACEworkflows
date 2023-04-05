
using Distributed 
addprocs(10)

@everywhere begin
   using Pkg; Pkg.activate(@__DIR__()); Pkg.instantiate()
   using ACE1pack, JuLIP, LinearAlgebra, PrettyTables
end

# the dataset is provided via ACE1pack artifacts as a convenient benchmarkset
datapath = joinpath(ACE1pack.artifact("ZuoEtAl2020"), "ZuoEtAl2020")
syms = [:Ni, :Cu, :Li, :Mo, :Si, :Ge]

totaldegree_sm = [ 20, 16, 12 ]   # small model: ~ 300  basis functions
totaldegree_lge = [ 25, 21, 17 ]  # large model: ~ 1000 basis functions              

## 

# for these problems, virtually all solvers give the same result 
# since the regularisation is all but irrelevant. 
# brr = ACEfit.RRQR(; rtol=1e-12)
# brr = ACEfit.BayesianLinearRegressionSVD()
solver = ACEfit.SKLEARN_BRR(; n_iter = 1000)

err = Dict("lge" => Dict("E" => Dict(), "F" => Dict()), 
            "sm" => Dict("E" => Dict(), "F" => Dict()) )

for sym in syms 
   @info("---------- fitting $(sym) ----------")
   train = JuLIP.read_extxyz(joinpath(datapath, "$(sym)_train.xyz"))
   test = JuLIP.read_extxyz(joinpath(datapath, "$(sym)_test.xyz"))

   # specify the models 
   model_lge = acemodel(elements = [sym,], order = 3, totaldegree = totaldegree_lge)
   model_sm = acemodel(elements = [sym,], order = 3, totaldegree = totaldegree_sm)
   @info("$sym models: length = $(length(model_lge.basis)), $(length(model_sm.basis))")

   # train the model 
   ACE1pack.acefit!(model_sm, train; solver=solver, mode=:distributed)
   ACE1pack.acefit!(model_lge, train; solver=solver, mode=:distributed)

   # compute and store errors for later visualisation
   err_sm  = ACE1pack.linear_errors(test,  model_sm)
   err_lge  = ACE1pack.linear_errors(test,  model_lge)
   err["sm" ]["E"][sym] =  err_sm["mae"]["set"]["E"] * 1000
   err["lge"]["E"][sym] = err_lge["mae"]["set"]["E"] * 1000
   err["sm" ]["F"][sym] =  err_sm["mae"]["set"]["F"]
   err["lge"]["F"][sym] = err_lge["mae"]["set"]["F"]
end


##

header = ([ "", "ACE(sm)", "ACE(lge)", "GAP", "MTP"])
e_table_gap_mtp = [ 0.42  0.48; 0.46  0.41; 0.49  0.49; 2.24  2.83; 2.91  2.21; 2.06  1.79]
f_table_gap_mtp = [ 0.02  0.01; 0.01  0.01; 0.01  0.01; 0.09  0.09; 0.07  0.06; 0.05  0.05]
      
e_table = hcat(string.(syms),
         [round(err["sm"]["E"][sym], digits=3) for sym in syms], 
         [round(err["lge"]["E"][sym], digits=3) for sym in syms],
         e_table_gap_mtp)
     
f_table = hcat(string.(syms),
         [round(err["sm"]["F"][sym], digits=3) for sym in syms], 
         [round(err["lge"]["F"][sym], digits=3) for sym in syms],
         f_table_gap_mtp)         

println("Energy Error")         
pretty_table(e_table; header = header)

println("Force Error")         
pretty_table(f_table; header = header)

##

pretty_table(e_table, backend = Val(:latex), label = "Energy MAE", header = header)
pretty_table(f_table, backend = Val(:latex), label = "Forces MAE", header = header)
