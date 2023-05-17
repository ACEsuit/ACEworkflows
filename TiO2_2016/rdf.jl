
using ACE1pack

data = read_extxyz(@__DIR__() * "/TiO2trainingset.xyz")

r_TiTi = Float64[] 
r_TiO = Float64[]
r_OO = Float64[]

rcut = 6.2

at = data[1]
nlist = neighborlist(at, rcut)
