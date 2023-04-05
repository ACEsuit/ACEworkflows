
using Distributed
addprocs(10)

@everywhere begin
   using Pkg; Pkg.activate(@__DIR__()); Pkg.instantiate()
   using ACE1pack, JuLIP, LinearAlgebra, PrettyTables
end
##


# download the datafile and replace the path 
datapath = "/Users/ortner/Dropbox/data/training_sets/silicon.xyz"
data = JuLIP.read_extxyz(datapath)

# for prototyping use the following parameters: 
solver = ACEfit.SKLEARN_BRR(; n_iter = 1000)
# totaldegree = [25, 20, 15, 10]

# for final fit: use larger model and ARD instead of BRR 
# solver = ACEfit.SKLEARN_ARD(; n_iter = 1000)
totaldegree = [26, 26, 23, 20]

## 

# construct the model with just the defaults, but 
E0 = -158.54496821
model = acemodel(elements = [:Si,], 
                 order = 4, 
                 totaldegree = totaldegree,
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

datakeys = (:energy_key => "dft_energy", 
           :force_key => "dft_force", 
           :virial_key => "dft_virial" )

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

# need to find the right weights 
ACE1pack.acefit!(model, data; solver=solver, mode=:distributed, 
                             weights=weights, datakeys...)

# this dataset doesn't have a test set, so we can only show the fit errors 
errors = ACE1pack.linear_errors(data, model; datakeys...)

