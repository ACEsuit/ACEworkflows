using ACEpotentials
model = acemodel(elements = [:Ti, :Al],
                 order = 4
                 totaldegree = [25, 23, 20, 10],
                 wL = 2.0)
