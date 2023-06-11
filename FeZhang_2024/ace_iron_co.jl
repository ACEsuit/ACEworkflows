using ACE1pack, Random, Distributed 
include(@__DIR__() * "/utils.jl")

addprocs(10)
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__())
@everywhere Pkg.instantiate()
@everywhere using ACE1pack, JuLIP, LinearAlgebra, PrettyTables

##


data_file = @__DIR__() * "/Fe_train.xyz"
data = read_extxyz(data_file)
numobs_estimate = 3 * sum(length, data)
@show numobs_estimate

datakeys = (energy_key = "energy", 
            force_key = "force", 
            virial_key = "virial")

# split training and testing sets -> 90:10
training_data, testing_data = train_test_split(data, 0.1)

##

r_cut=6.5

# try 200 to 3000 parameters 
# try range of transforms and cutoffs 

# totaldegree = [ 20, 16, 10 ]
# totaldegree = [ 24, 19, 14 ]
totaldegree = [ 27, 22, 17 ]
# totaldegree = [ 27, 22, 17, 12 ]
# totaldegree = [ 28, 23, 18, 12 ]
# totaldegree = [ 30, 25, 20, 13 ]
# totaldegree = [ 32, 27, 22, 14 ]

model = acemodel(elements = [:Fe,], order = length(totaldegree), 
                 totaldegree = totaldegree, rcut = r_cut, 
                 transform = (:agnesi, 1, 2), 
                 E0 = Dict(:Fe => -3455.6995339) )

basis = model.basis                  
@show length(basis)

##

weights = Dict(
    "default" => Dict("E" => 12.5, "F" => 1.0 , "V" => 0.0 ),
   "slice_sample_high" => Dict("E" => 100.0, "F" => 0.0 , "V" => 0.01 ),
   "phonons_54_high" => Dict("E" => 46.0, "F" => 1.0 , "V" => 0.0 ),
   "prim_random" => Dict("E" => 100.0, "F" => 0.0 , "V" => 0.01 ),    
   "phonons_128_high" => Dict("E" => 19.0, "F" => 1.0 , "V" => 0.0 )
   )


# acefit!(model, training_data; 
#        mode = :distributed, solver = solver, weights = weights, datakeys...)

_train = [ AtomsData(at; weights=weights, v_ref = model.Vref, datakeys...)
           for at in train_data ]

A, Y, W = ACEfit.linear_assemble(_train, model.basis, :distributed)

##

solver = ACEfit.RRQR(rtol = 1e-12)
P = ACE1pack._make_prior(model, 2, nothing)
Ap = Diagonal(W) * (A / P) 
Yp = W .* Y

result = ACEfit.linear_solve(solver, Ap, Yp)
c = P \ result["C"]
ACE1pack._set_params!(model, c)

##   
     
@info("TRAINING ERRORS: ")
ACE1pack.linear_errors(training_data, model; datakeys...)

@info("TEST ERRORS: ")
ACE1pack.linear_errors(testing_data, model; datakeys...)

##

# save_dict("./Fe_ace_N$(N)_D$(D)_R$(r_cut).json", Dict("IP" => write_dict(IP), "info" => lsqinfo))
# ACE1pack.ExportMulti.export_ACE("./Fe_ace_N$(N)_D$(D)_R$(r_cut).yace", IP)
