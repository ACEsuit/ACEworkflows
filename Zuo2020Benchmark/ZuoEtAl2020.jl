
using Distributed 
addprocs(10)

@everywhere begin
   using Pkg
   Pkg.activate(@__DIR__())
   Pkg.instantiate()
   using ACE1pack, JuLIP, LinearAlgebra, PrettyTables
end


datapath = joinpath(ACE1pack.artifact("ZuoEtAl2020"), "ZuoEtAl2020")
syms = [:Ni, :Cu, :Li, :Mo, :Si, :Ge]
rcuts = Dict(:Ni => 4.0, :Cu => 3.9, :Li => 5.1, 
             :Mo => 5.2, :Si => 4.7, :Ge => 5.1)

## 

e_train_brr = Dict() 
e_test_brr = Dict()
f_train_brr = Dict()
f_test_brr = Dict()

_ard = false 
e_train_ard = Dict() 
e_test_ard = Dict()
f_train_ard = Dict()
f_test_ard = Dict()

for sym in syms 
   # rcut = 2 * rnn(sym)
   rcut = 2.5 * rnn(sym)
   # rcut = rcuts[sym] 
   @info("---------- fitting $(sym) ----------")

   train = JuLIP.read_extxyz(joinpath(datapath, "$(sym)_train.xyz"))
   test = JuLIP.read_extxyz(joinpath(datapath, "$(sym)_test.xyz"))
   # train = train[1:5:end]

   model = acemodel(elements = [sym,],
                    order = 3, 
                    totaldegree = [25, 22, 17], 
                    rcut = rcut, )

   @show length(model.basis)

   function test_errors(model)
      _test = [ AtomsData(at, "energy", "force", "virial", 
                           weights, model.Vref) for at in test ]
      test_rmse, test_mae = ACE1pack.linear_errors(test, model)
      return test_mae[2]["set"]
   end

   weights = Dict("default" => Dict("E" => 30.0, "F" => 1.0))
   _train = [ AtomsData(at, "energy", "force", "virial", 
                     weights, model.Vref) for at in train ]
   A, Y, W = ACEfit.linear_assemble(_train, model.basis, :distributed)
   P = ACE1pack._make_prior(model, 2, nothing)
   Ap = Diagonal(W) * (A / P) 
   Y = W .* Y

   # BRR solver 
   # brr = ACEfit.RRQR(; rtol=1e-12)
   # brr = ACEfit.BayesianLinearRegressionSVD()
   brr = ACEfit.SKLEARN_BRR(; n_iter = 1000)
   result_brr = ACEfit.linear_solve(brr, Ap, Y)

   coeffs = P \ result_brr["C"]
   ACE1x._set_params!(model, coeffs)
   brr_train_rmse, brr_train_mae = ACE1pack.linear_errors(train, model)
   brr_test_rmse, brr_test_mae = ACE1pack.linear_errors(test, model)

   e_train_brr[sym] = brr_train_mae[2]["set"]["E"] * 1000
   e_test_brr[sym] = brr_test_mae[2]["set"]["E"] * 1000
   f_train_brr[sym] = brr_train_mae[2]["set"]["F"]
   f_test_brr[sym] = brr_test_mae[2]["set"]["F"]

   # ARD solver 
   if _ard 
      ard = ACEfit.SKLEARN_ARD(; n_iter=1000)
      result_ard = ACEfit.linear_solve(ard, Ap, Y)

      coeffs = P \ result_ard["C"]
      ACE1x._set_params!(model, coeffs)
      ard_train_rmse, ard_train_mae = ACE1pack.linear_errors(train, model)
      ard_test_rmse, ard_test_mae = ACE1pack.linear_errors(test, model)

      e_train_ard[sym] = ard_train_mae[2]["set"]["E"] * 1000
      e_test_ard[sym] = ard_test_mae[2]["set"]["E"] * 1000
      f_train_ard[sym] = ard_train_mae[2]["set"]["F"]
      f_test_ard[sym] = ard_test_mae[2]["set"]["F"]
   else 
      e_train_ard[sym] = NaN
      e_test_ard[sym] = NaN
      f_train_ard[sym] = NaN
      f_test_ard[sym] = NaN
   end 
end


##

header = ([ "", "ACE(BRR)", "ACE(BRR)", "ACE(ARD)", "ACE(ARD)", "GAP", "GAP", "MTP", "MTP"],
          [ "", "train", "test", "train", "test", "train", "test", "train", "test"])

e_table_gap_mtp = [ 0.42  0.42  0.45  0.48;
                    0.48  0.46  0.4   0.41;
                    0.55  0.49  0.49  0.49;
                    2.08  2.24  2.43  2.83;
                    3.25  2.91  2.05  2.21;
                    2.15  2.06  1.86  1.79]

f_table_gap_mtp = [ 0.01  0.02  0.01  0.01;
                    0.01  0.01  0.01  0.01;
                    0.01  0.01  0.01  0.01;
                    0.08  0.09  0.08  0.09;
                    0.06  0.07  0.04  0.06;
                    0.04  0.05  0.04  0.05]
      
e_table = hcat(string.(syms),
         [round(e_train_brr[sym], digits=3) for sym in syms], 
         [round(e_test_brr[sym], digits=3) for sym in syms],
         [round(e_train_ard[sym], digits=3) for sym in syms], 
         [round(e_test_ard[sym], digits=3) for sym in syms],
         e_table_gap_mtp)
     
f_table = hcat(string.(syms),
         [round(f_train_brr[sym], digits=3) for sym in syms], 
         [round(f_test_brr[sym], digits=3) for sym in syms],
         [round(f_train_ard[sym], digits=3) for sym in syms], 
         [round(f_test_ard[sym], digits=3) for sym in syms],
         f_table_gap_mtp)         

##
println("Energy Error")         
pretty_table(e_table; header = header)

println("Force Error")         
pretty_table(f_table; header = header)

##

# pretty_table(e_table_brr, backend = Val(:latex), label = "Energy MAE", header = header)
# pretty_table(f_table_brr, backend = Val(:latex), label = "Forces MAE", header = header)
