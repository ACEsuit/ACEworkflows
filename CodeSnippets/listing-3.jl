include("listing-2.jl")

#model = ... # cf. Listing 2
P = smoothness_prior(model)
data, _, _ = ACEpotentials.example_dataset("TiAl_tutorial")
acefit!(model, data; prior = P, solver = ACEfit.BLR())
export2lammps("TiAl.yace", model)
