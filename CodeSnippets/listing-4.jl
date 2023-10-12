using ACEpotentials
elements = [:Ti, :Al]
totaldegree = 12
r0 = (rnn(:Ti) + rnn(:Al)) / 2
rcut = 2 * r0
trans = AgnesiTransform(; r0=r0, p = 2)
fenv = PolyEnvelope(1, r0, rcut)
radbasis = transformed_jacobi_env(totaldegree, trans, fenv, rcut)
model = acemodel(elements = elements,
                 order = 3,
                 totaldegree = totaldegree,
                 radbasis = radbasis)
