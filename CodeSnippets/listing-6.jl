include("listing-2.jl")
#model = ... # cf. Listing 2
Γ = smoothness_prior(model; p = 2, wL = 1)
