#     rin = 0.5; rcut = 5.44923234640252;
#     acedescriptors[1] = ACEpot.ACEparams(species = [:Ti,:O], nbody=1, pdegree=6, r0=ACEpot.rnn(:Ti), rcut=rcut, rin=rin, wL=2.0, csp=1.75)
#     acedescriptors[2] = ACEpot.ACEparams(species = [:Ti,:O], nbody=2, pdegree=12, r0=ACEpot.rnn(:Ti), rcut=rcut, rin=rin, wL=1.5, csp=1.5)
#     acedescriptors[3] = ACEpot.ACEparams(species = [:Ti,:O], nbody=4, pdegree=12, r0=ACEpot.rnn(:Ti), rcut=rcut, rin=rin, wL=1.35, csp=1.25)

# budget : 870 basis functions!!

using ACE1pack, Distributed
addprocs(10)

@everywhere using Pkg 
@everywhere  Pkg.activate(@__DIR__());
@everywhere  Pkg.instantiate()
@everywhere  using ACE1pack, Distributed

## load the data 

data = read_extxyz(@__DIR__() * "/TiO2trainingset.xyz")
traindata = data[1:2:end]
testdata = data[2:2:end]

datakeys = (:energy_key => "energy",
           :force_key => "forces",
           :virial_key => "virial" )
           

# estimate the number of observations : this is a huge dataset, what for?!
num_obs = sum( ((length(at)-1)*3 + 1) for at in traindata )
@show num_obs

## generate a model 

# totaldegree = [22, 18, 13, 9]   # ~ 5000
totaldegree = [20, 16, 12]   # ~ 3000 
# totaldegree = [20, 12, 8]   # ~ 870

Eref = Dict(:Ti => -1609.1351182958786, 
             :O => -434.97357935367666)


model = acemodel(elements = [:Ti, :O], 
                 order = length(totaldegree), 
                 totaldegree = totaldegree, 
                 wL = 1.35, 
                 rcut = 6.2, 
                 Eref = Eref, 
                 )
@show length(model.basis)

## estimate the parameters 


acefit!(model, traindata; 
       solver=ACEfit.RRQR(;rtol = 1e-7), 
       mode=:distributed, 
       datakeys...)

## analyze the errors 

@info("Error on training data")
errors = ACE1pack.linear_errors(traindata, model; datakeys...)
@info("Error on test data")
errors = ACE1pack.linear_errors(testdata, model; datakeys...)


# old MAE 
# [ Info: MAE Table
# ┌─────────┬─────────┬──────────┬─────────┐
# │    Type │ E [meV] │ F [eV/A] │ V [meV] │
# ├─────────┼─────────┼──────────┼─────────┤
# │ default │   4.913 │    0.206 │   0.000 │
# ├─────────┼─────────┼──────────┼─────────┤
# │     set │   4.913 │    0.206 │   0.000 │
# └─────────┴─────────┴──────────┴─────────┘

# tiny model: 
# [ Info: MAE Table
# ┌─────────┬─────────┬──────────┬─────────┐
# │    Type │ E [meV] │ F [eV/A] │ V [meV] │
# ├─────────┼─────────┼──────────┼─────────┤
# │ default │  19.908 │    0.443 │   0.000 │
# ├─────────┼─────────┼──────────┼─────────┤
# │     set │  19.908 │    0.443 │   0.000 │
# └─────────┴─────────┴──────────┴─────────┘

# medium-scale model with 4800 basis functions 
# [ Info: MAE Table
# ┌─────────┬─────────┬──────────┬─────────┐
# │    Type │ E [meV] │ F [eV/A] │ V [meV] │
# ├─────────┼─────────┼──────────┼─────────┤
# │ default │   2.414 │    0.100 │   0.000 │
# ├─────────┼─────────┼──────────┼─────────┤
# │     set │   2.414 │    0.100 │   0.000 │
# └─────────┴─────────┴──────────┴─────────┘
