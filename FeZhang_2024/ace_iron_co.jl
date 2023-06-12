using ACE1pack, JuLIP, Random, Distributed, 
      PrettyTables, LinearAlgebra, HDF5, Lasso 

addprocs(64)
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__())
@everywhere Pkg.instantiate()
@everywhere using ACE1pack, JuLIP

##

@info("Loading data...")
training_data = read_extxyz(@__DIR__() * "/Fe_train.xyz")
testing_data = read_extxyz(@__DIR__() * "/Fe_test.xyz")
numobs_estimate = 3 * sum(length, training_data)
@show numobs_estimate

datakeys = (energy_key = "energy", 
            force_key = "force", 
            virial_key = "virial")

##

@info("Setting up model...")
r_cut=6.5

# try 200 to 3000 parameters 
# try range of transforms and cutoffs 

# totaldegree = [ 20, 16, 10 ]
# totaldegree = [ 24, 19, 14 ]
# totaldegree = [ 27, 22, 17 ]
# totaldegree = [ 27, 22, 17, 12 ]
# totaldegree = [ 28, 23, 18, 12 ]
# totaldegree = [ 30, 25, 20, 13 ]
# totaldegree = [ 32, 27, 22, 14 ]
totaldegree = [ 33, 28, 23, 18 ]

model = acemodel(elements = [:Fe,], order = length(totaldegree), 
                 totaldegree = totaldegree, rcut = r_cut, 
                 transform = (:agnesi, 1, 2), 
                 E0 = Dict(:Fe => -3455.6995339) )

basis = model.basis                  
@show length(basis)

##

@info("Assemble design matrix...")

weights = Dict(
    "default" => Dict("E" => 12.5, "F" => 1.0 , "V" => 0.0 ),
   "slice_sample_high" => Dict("E" => 100.0, "F" => 0.0 , "V" => 0.01 ),
   "phonons_54_high" => Dict("E" => 46.0, "F" => 1.0 , "V" => 0.0 ),
   "prim_random" => Dict("E" => 100.0, "F" => 0.0 , "V" => 0.01 ),    
   "phonons_128_high" => Dict("E" => 19.0, "F" => 1.0 , "V" => 0.0 )
   )

_train = [ AtomsData(at; weights=weights, v_ref = model.Vref, datakeys...)
           for at in training_data ]

potid = prod("_" .* string.(totaldegree))
path_lsq_sys = @__DIR__() * "/fe_lsqsys" * potid * ".h5"

if !isfile(path_lsq_sys)
   A, Y, W = ACEfit.linear_assemble(_train, model.basis, :distributed)

   h5open(path_lsq_sys, "w") do file
      write(file, "A", A)
      write(file, "Y", Y)
      write(file, "W", W)
   end
end 

##

# load the pre-assembled design matrix 
A, Y, W = h5open(path_lsq_sys, "r") do file
   read(file, "A", "Y", "W")
end

## 
# compute a lasso path 

P = ACE1pack._make_prior(model, 2, nothing)
Ap = Diagonal(W) * (A / P) 
Yp = W .* Y

# lassopath = fit(LassoPath, Ap, Yp, 
#                standardize = false, intercept = false,
#                λminratio=1e-6, randomize=false, stopearly=false, 
#                nλ = 1000, 
#                cd_tol=1e-5)


##

solver = ACEfit.RRQR(rtol = 1e-12)
result = ACEfit.linear_solve(solver, Ap, Yp)
c = P \ result["C"]
ACE1pack._set_params!(model, c)

##   
     
@info("TRAINING ERRORS: ")
train_errs = ACE1pack.linear_errors(training_data, model; datakeys...)

@info("TEST ERRORS: ")
test_errs = ACE1pack.linear_errors(testing_data, model; datakeys...)

##

# save_dict("./Fe_ace_N$(N)_D$(D)_R$(r_cut).json", Dict("IP" => write_dict(IP), "info" => lsqinfo))
# ACE1pack.ExportMulti.export_ACE("./Fe_ace_N$(N)_D$(D)_R$(r_cut).yace", IP)
