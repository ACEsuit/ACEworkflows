@info "BEGIN LISTING 1"

using ACEpotentials
data, _, _ = ACEpotentials.example_dataset("TiAl_tutorial")
model = acemodel(elements = [:Ti, :Al],
                 order = 3,
                 totaldegree = 12,
                 Eref = [:Ti => -1586.0195, :Al => -105.5954])
acefit!(model, data)
export2lammps("TiAl_tutorial.yace", model)
