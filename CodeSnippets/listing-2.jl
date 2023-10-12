using ACEpotentials
model = acemodel(; elements = [:Ti, :Al],
                   order = 3,
                   totaldegree = 12,
                   rcut = 5.5,
                   Eref = [:Ti => -1586.0195, :Al => -105.5954])
