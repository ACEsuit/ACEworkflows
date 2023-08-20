using ACEpotentials
model = acemodel(elements = [:Ti, :Al],
                 order = 4,
                 wL = 2.0,
                 totaldegree = [25, 23, 20, 10])
