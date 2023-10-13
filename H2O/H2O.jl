
using Distributed, HDF5
addprocs(10)

@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__())
@everywhere Pkg.instantiate()
@everywhere using ACE1pack, LinearAlgebra, PrettyTables

##

# download the datafile and replace the path 
# https://files.slack.com/files-pri/T53DERT7A-F0567DSRSLA/download/dataset_1593_evang.xyz?origin_team=T53DERT7A
datapath = @__DIR__() * "/H2O_eVAng.xyz"
data = read_extxyz(datapath)

datakeys = (energy_key = "TotEnergy", 
            force_key = "force", 
            virial_key = "" )

# use some typical defaults. this dataset seems to have no configtype
weights = Dict(
    "default" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 ),
    # "dia" => Dict("E" => 60.0, "F" => 2.0 , "V" => 2.0 ),
  )

numobs = sum( (3*length(at) + 1)  for at in data )
@show numobs 

## 

# model degree parameters, make sure those don't change between runs 
# so that the design matrix is consistent with the model!
# this is a bit delicate and should probably be done in a more robust way.

# for prototyping use the following parameters: 
# totaldegree = [18, 12, 6]
totaldegree = [20, 15, 10]
# totaldegree = [25, 16, 12]
# totaldegree = [25, 18, 14, 10]

# molecular distance is ca 0.3nm = 3A 
# H bonds are ca 0.1nm = 1A 
# so important bonding happens on quite a large scale. This is a paradigm case 
# for a multi-scale basis, but we don't have time for that now ... Instead we 
# will use an agnesi-2-1, but with custom r0 values that capture the typical 
# lengths of the HH, HO bonds and the average OO distance in water. 

E0 = Dict("H" => -187.60433405933455, "O" => -93.80216702966727)
r0 = Dict((:H, :O) => 1.0, (:O, :H) => 1.0, (:H, :H) => 2.4, (:O, :O) => 3.0)

rcut = 6.0 # only try to see the nearest molecule but not the next one. 

# maybe for HH we should use > 2.4 up to 2.8 which is the intermolecular distance

# construct the model with just the defaults, but 
model = acemodel(elements = [:H, :O], 
                 order = length(totaldegree), 
                 transform = (:agnesi, 1, 2), 
                 rcut = rcut, 
                 r0 = r0, 
                 totaldegree = totaldegree,
                 E0 = E0)

@info("ACE Si Model with length = $(length(model.basis))")

##

# For such a big dataset it might be best to pre-assemble the design matrix 
_data = data[1:2:end]  # replace with suitable subset ... 
_train = [ AtomsData(at; datakeys..., weights=weights, v_ref = model.Vref) for at in _data ]

pot_id = prod("_" .* string.(totaldegree))
path_lsq_sys = @__DIR__() * "/H2O_lsqsys" * pot_id * ".h5"                     

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

# solver = ACEfit.RRQR(rtol = 1e-6)
solver = ACEfit.SKLEARN_BRR(; n_iter = 1000)
# solver = ACEfit.SKLEARN_ARD(; n_iter = 1000)

P = ACE1pack._make_prior(model, 2, nothing)
Ap = Diagonal(W) * (A / P) 
Yp = W .* Y

result = ACEfit.linear_solve(solver, Ap, Yp)
cp = result["C"]
c = Diagonal(P) \ cp

ACE1pack._set_params!(model, c)

@info("ERRORS - TRAINING SET")
train_errors = ACE1pack.linear_errors(_data, model; datakeys...)


# let's pick out a test set 
@info("ERRORS - TEST SET")
testset = data[2:2:end]
test_errors = ACE1pack.linear_errors(data, model; datakeys...) 


## save potential as JSON for Molly 
potpath = @__DIR__() * "/H2O" * pot_id * ".json"
export2json(potpath, model)

# # load the potential and confirm it is consistent :) 
# potential = read_dict(load_json(potpath)["potential"])
# all( energy(model.potential, at) ≈ energy(potential, at) for at in rand(data, 30) )