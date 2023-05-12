
using Distributed, HDF5
addprocs(64)

@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__())
@everywhere Pkg.instantiate()
@everywhere using ACE1pack, LinearAlgebra, PrettyTables


##

# download the datafile and replace the path 
datapath = "/Users/ortner/Dropbox/data/training_sets/silicon.xyz"
# datapath = "/zfs/users/ortner/ortner/datasets/potentials/silicon.xyz"
data = read_extxyz(datapath)

# filter for an easy initial fit? - remove this filter for final results ... 
cfgtypes = ["dia", "amorph"]
# cfgtype = ["dia", "amorph", "vacancy", "divacancy", "bcc", "fcc"] 
data = filter(at -> get_data(at, "config_type") in cfgtypes, data)

datakeys = (energy_key = "dft_energy", 
            force_key = "dft_force", 
            virial_key = "dft_virial" )

numobs_estimate = 3 * sum(length.(data))
@show numobs_estimate

## 

# model degree parameters, make sure those don't change between runs 
# so that the design matrix is consistent with the model!
# this is a bit delicate and should probably be done in a more robust way.

# models of different sizes 
totaldegree = [20, 15, 10]         # 202
# totaldegree = [25, 20, 15, 10]     # ~ 715
# totaldegree = [25, 22, 18, 13]     # ~ 1500 
# totaldegree = [28, 26, 21, 16]     # ~ 3300
# totaldegree = [28, 28, 24, 20]     # ~ 8200

# construct the model with just the defaults, but 
E0 = -158.54496821
model = acemodel(elements = [:Si,], 
                 order = length(totaldegree), 
                 totaldegree = totaldegree,
                 transform = (:agnesi, 1, 2), 
                 E0 = E0)

@info("ACE Si Model with length = $(length(model.basis))")

##

# before we fit we have to fix the up the data 
# we remove the first one which just contains the isolated atom 
# with energy E0 
if data[1].data["dft_energy"].data ≈ E0
   deleteat!(data, 1)
end

##


# these are the weights used in Lysorgsky et al 2020 
weights = Dict(
    "default" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 ),
    #"dia" => Dict("E" => 60.0, "F" => 2.0 , "V" => 2.0 ),
    "2b_dimer" => Dict("E" => 0.0, "F" => 0.0 , "V" => 0.0 ),
    "liq" => Dict("E" => 10.0, "F" => 0.66, "V" => 0.25 ),
    "amorph" => Dict("E" => 3.0, "F" => 0.5 , "V" => 0.1),
    "sp" => Dict("E" => 3.0, "F" => 0.5 , "V" => 0.1),
    "bc8"=> Dict("E" => 50.0, "F" => 0.5 , "V" => 0.1),
    "vacancy" => Dict("E" => 50.0, "F" => 0.5 , "V" => 0.1), 
    "interstitial" => Dict("E" => 50.0, "F" => 0.5 , "V" => 0.1), 
  )


##

# For such a big dataset it might be best to pre-assemble the design matrix 
_data = data[1:2:end]
_train = [ AtomsData(at; weights=weights, v_ref = model.Vref, datakeys...)
           for at in _data ]

potid = prod("_" .* string.(totaldegree))
path_lsq_sys = @__DIR__() * "/si_lsqsys" * potid * ".h5"

# to re-assemble the design matrix, delete the file
if !isfile(path_lsq_sys)
   A, Y, W = ACEfit.linear_assemble(_train, model.basis, :distributed)

   h5open(path_lsq_sys, "w") do file
      write(file, "A", A)
      write(file, "Y", Y)
      write(file, "W", W)
   end
end 

##

# this could be in a separate script, but for simplicity I'll keep it in 
# one place for now ... 

# load the pre-assembled design matrix 
A, Y, W = h5open(path_lsq_sys, "r") do file
   read(file, "A", "Y", "W")
end

# redefine and recompute the weights as needed 

weights["dia"] = Dict("E" => 20.0, "F" => 0.5, "V" => 0.5 )
W = ACE1pack.recompute_weights(model, _data;
                weights=weights, datakeys...)


# I get very good test errors even with tiny rtol ... 
solver = ACEfit.RRQR(rtol = 1e-12)
# BRR gives complete CRAP for this - no idea why?!?
# solver = ACEfit.SKLEARN_BRR(; n_iter = 1000)
# solver = ACEfit.SKLEARN_ARD(; n_iter = 1000)

P = ACE1pack._make_prior(model, 2, nothing)
Ap = Diagonal(W) * (A / P) 
Yp = W .* Y

result = ACEfit.linear_solve(solver, Ap, Yp)
cp = result["C"]
c = P \ cp

ACE1pack._set_params!(model, c)

# this dataset doesn't have a test set, so we can only show the fit errors 
@info("TRAINING ERRORS")
errors = ACE1pack.linear_errors(_data, model; datakeys...)


# # test set  -  to use for preliminary testing and fine-tuning
# @info("TEST ERRORS")
# testdata = data[2:2:end]
# testerrors = ACE1pack.linear_errors(testdata, model; datakeys...)
